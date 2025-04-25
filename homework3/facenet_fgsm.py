# 导入facenet的加载模块
import facenet_fn
import tensorflow as tf
import numpy as np

# 引入CleverHans的Model抽象类和FGSM攻击方法
from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod

# 自定义的测试集加载模块
import set_loader


# 定义用于封装Facenet模型的类
class InceptionResnetV1Model(Model):
    # 模型的预训练权重文件路径
    model_path = "models/20180402-114759/20180402-114759.pb"

    def __init__(self):
        # 调用父类构造方法，定义作用域
        super(InceptionResnetV1Model, self).__init__(scope="model")

        # 加载Facenet预训练模型
        facenet_fn.load_model(self.model_path)

        # 获取当前默认计算图
        graph = tf.get_default_graph()

        # 提取输入张量（图片输入）
        self.face_input = graph.get_tensor_by_name("input:0")
        # 提取输出张量（512维的特征嵌入）
        self.embedding_output = graph.get_tensor_by_name("embeddings:0")

    def convert_to_classifier(self):
        # 定义一个占位符用于输入目标图像的特征向量（嵌入）
        self.victim_embedding_input = tf.placeholder(tf.float32, shape=(None, 512))

        # 计算输入图像和目标图像嵌入之间的欧氏距离
        distance = tf.reduce_sum(
            tf.square(self.embedding_output - self.victim_embedding_input), axis=1
        )

        # 将欧氏距离转换为“分类概率”形式（模拟分类器）
        threshold = 0.99  # Facenet默认相似度阈值
        score = tf.where(
            distance > threshold,
            0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
            0.5 * distance / threshold,
        )
        reverse_score = 1.0 - score

        # 构造softmax格式输出（2分类）
        self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))

        # 保存softmax输出作为“logits”层
        self.layer_names = []
        self.layers = []
        self.layers.append(self.softmax_output)
        self.layer_names.append("logits")

    # CleverHans要求实现fprop接口返回层字典
    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))


# 创建默认计算图
with tf.Graph().as_default():
    with tf.Session() as sess:
        # 实例化模型并转换为分类器结构
        model = InceptionResnetV1Model()
        model.convert_to_classifier()

        # 加载用于测试的人脸图像对，以及对应的标签（0-同人，1-不同人）
        size = 100  # 加载100对
        faces1, faces2, labels = set_loader.load_testset(size)

        # 使用facenet模型提取faces2的特征嵌入，作为攻击目标
        graph = tf.get_default_graph()
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
        feed_dict = {model.face_input: faces2, phase_train_placeholder: False}
        victims_embeddings = sess.run(model.embedding_output, feed_dict=feed_dict)

        # 定义FGSM攻击器参数
        steps = 1
        eps = 0.01  # 总扰动大小
        alpha = eps / steps  # 每步扰动（这里只有一步）
        fgsm = FastGradientMethod(model)
        fgsm_params = {"eps": alpha, "clip_min": 0.0, "clip_max": 1.0}
        adv_x = fgsm.generate(model.face_input, **fgsm_params)

        # 执行FGSM攻击
        adv = faces1  # 初始化为原始图像
        for i in range(steps):
            print("FGSM step " + str(i + 1))
            feed_dict = {
                model.face_input: adv,
                model.victim_embedding_input: victims_embeddings,
                phase_train_placeholder: False,
            }
            adv = sess.run(adv_x, feed_dict=feed_dict)

        # 测试模型在原始图像上的识别准确率
        batch_size = graph.get_tensor_by_name("batch_size:0")
        feed_dict = {
            model.face_input: faces1,
            model.victim_embedding_input: victims_embeddings,
            phase_train_placeholder: False,
            batch_size: 64,
        }
        real_labels = sess.run(model.softmax_output, feed_dict=feed_dict)
        accuracy = np.mean(
            (np.argmax(labels, axis=-1)) == (np.argmax(real_labels, axis=-1))
        )
        print("Accuracy: " + str(accuracy * 100) + "%")

        # 测试模型在对抗样本上的识别准确率
        feed_dict = {
            model.face_input: adv,
            model.victim_embedding_input: victims_embeddings,
            phase_train_placeholder: False,
            batch_size: 64,
        }
        adversarial_labels = sess.run(model.softmax_output, feed_dict=feed_dict)

        # 计算同一个人的对抗成功率（逃避攻击）
        same_faces_index = np.where((np.argmax(labels, axis=-1) == 0))
        accuracy = np.mean(
            (np.argmax(labels[same_faces_index], axis=-1))
            == (np.argmax(adversarial_labels[same_faces_index], axis=-1))
        )
        print(
            "Accuracy against adversarial examples for "
            + "same person faces (dodging): "
            + str(accuracy * 100)
            + "%"
        )

        # 计算不同人的对抗成功率（冒充攻击）
        different_faces_index = np.where((np.argmax(labels, axis=-1) == 1))
        accuracy = np.mean(
            (np.argmax(labels[different_faces_index], axis=-1))
            == (np.argmax(adversarial_labels[different_faces_index], axis=-1))
        )
        print(
            "Accuracy against adversarial examples for "
            + "different people faces (impersonation): "
            + str(accuracy * 100)
            + "%"
        )

        # 保存原图和对抗样本图像
        set_loader.save_images(adv, faces1, faces2, size)
