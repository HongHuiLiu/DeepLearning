图像分类/物体识别
第一部分：能够识别1000个对象的tensorflow图像分类器
第二部分：创建并训练一个全新的图像分类器
				第一部分
1、运行脚本找到最佳预测
python classifier.py --image_file 图片位置
2、运行脚本获得top n识别分类
python classifier.py --image_file 图片位置 --num_top_predictions n

				第二部分
1、设置图片文件夹
1）为每种类别创建一个文件夹，文件夹的名称就是类型的名称
2）将同一类别的所有图像添加到各自的文件夹中
3）将所有的文件夹，添加到一个父文件夹中
如：
~/flowers
 
~/flowers/roses/img1.jpg
 
~/flowers/roses/img2.jpg
 
...
 
~/flowers/tulips/tulips_img1.jpg
 
~/flowers/tulips/tulips_img2.jpg
 
~/flowers/tulips/tulips_img3.jpg
 
...
2、运行训练脚本
python retrain.py --model_dir ./inception --image_dir ~/flowers --output_graph ./output --how_many_training_steps 500
3、测试训练模型
python retrain_model_classifier.py 图片位置

参数说明：
--image_dir 标签图像文件夹的路径（目录下里面不能有其他图片！！）
--output_graph 训练的图像保存的位置
--output_labels 训练的图像的标签保存的位置
--summaries_dir TensorBoard的日志摘要的保存位置
--how_many_training_steps 训练结束前运行的训练步数
--learning_rate训练时使用的学习率大小
--testing_percentage 使用图像作为测试集的百分比
--validation_percentage使用图像作为验证集的百分比
--eval_step_interval 训练结果评估的时间间隔
--train_batch_size 一次训练的图像的数量
--test_batch_size 测试图像的数量。此测试集仅使用一次，以评估训练完成后模型的最终精度。值为-1时使用整个测试集，会在运行时得到更稳定结果。
--validation_batch_size在评价批次中使用的图像数量。此验证集比测试集使用得多，是模型在训练过程中准确度如何的一个早期的指标。值为-1时使用整个验证集，从而在训练迭代时得到更稳定的结果，但在大的训练集中可能会变慢。
--print_misclassified_test_images是否打印输出所有错误分类的测试图像列表。
--model_dir classify_image_graph_def.pb,imagenet_synset_to_human_label_map.txt和imagenet_2012_challenge_label_map_proto.pbtxt的路径
--bottleneck_dir 缓存的瓶颈层值的文件路径
--final_tensor_name 在重新训练的图像中输出的分类层的名字
--random_brightness训练图像输入像素上下的随机百分比大小

改善参数可以提高准确度！！
--flip_left_right是否随机水平翻转训练图像的一半
--random_crop 训练图像随机修剪的边缘百分比大小
--random_scale 训练图像随机缩放的尺寸百分比大小

