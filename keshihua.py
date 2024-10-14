# import cv2
# import os
#
# def txtShow(img, txt, save=True):
#     image = cv2.imread(img)
#     height, width = image.shape[:2]  # 获取原始图像的高和宽
#
#     # 读取classes类别信息
#     classes = ['Person', 'car', 'bicycle']
#     #
#
#     # 读取yolo格式标注的txt信息
#     with open(txt, 'r') as f:
#         labels = f.read().splitlines()
#     # ['0 0.403646 0.485491 0.103423 0.110863', '1 0.658482 0.425595 0.09375 0.099702', '2 0.482515 0.603795 0.061756 0.045387', '3 0.594122 0.610863 0.063244 0.052083', '4 0.496652 0.387649 0.064732 0.049107']
#
#     ob = []  # 存放目标信息
#     for i in labels:
#         cl, x_centre, y_centre, w, h = i.split(' ')
#
#         # 需要将数据类型转换成数字型
#         cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w), float(h)
#
#         name = classes[cl]  # 根据classes文件获取真实目标
#         xmin = int(x_centre * width - w * width / 2)  # 坐标转换
#         ymin = int(y_centre * height - h * height / 2)
#         xmax = int(x_centre * width + w * width / 2)
#         ymax = int(y_centre * height + h * height / 2)
#
#         tmp = [name, xmin, ymin, xmax, ymax]  # 单个检测框
#         ob.append(tmp)
#
#     # 绘制检测框
#     for name, x1, y1, x2, y2 in ob:
#         cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # 绘制矩形框
#         cv2.putText(image, name, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.5, thickness=1, color=(0, 0, 255))
#
#         # 保存图像
#     if save:
#         cv2.imwrite('result.png', image)
#
#         # 展示图像
#     cv2.imshow('test', image)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     lst_img=[]
#     lst_label=[]
#     img_path_fold = '/media/btbu/gt/ljx/FLIR-align-3class/FLIR-align-3class/infrared/val/images'  # 传入图片
#     label_path_fold = '/media/btbu/gt/ljx/FLIR-align-3class/FLIR-align-3class/visible/val/labels'  # 自动获取相应的txt标签文件
#
#     for i in os.listdir(img_path_fold):
#         img_path=img_path_fold+'/'+i
#         lst_img.append(img_path)
#     for i in os.listdir(label_path_fold):
#         label_path=label_path_fold+'/'+i
#         lst_label.append(label_path)
#     for i in range(len(lst_img)):
#         txtShow(img=lst_img[i], txt=lst_label[i], save=False)





import cv2


def txtShow(img, txt, save=True):
    image = cv2.imread(img)
    height, width = image.shape[:2]  # 获取原始图像的高和宽

    # 读取classes类别信息
    classes = ['Person', 'car', 'bicycle']
    #

    # 读取yolo格式标注的txt信息
    with open(txt, 'r') as f:
        labels = f.read().splitlines()
    # ['0 0.403646 0.485491 0.103423 0.110863', '1 0.658482 0.425595 0.09375 0.099702', '2 0.482515 0.603795 0.061756 0.045387', '3 0.594122 0.610863 0.063244 0.052083', '4 0.496652 0.387649 0.064732 0.049107']

    ob = []  # 存放目标信息
    for i in labels:
        cl, x_centre, y_centre, w, h,con = i.split(' ')

        # 需要将数据类型转换成数字型
        cl, x_centre, y_centre, w, h ,con= int(cl), float(x_centre), float(y_centre), float(w), float(h),float(con)
        if con>0.6:
            name = classes[cl]  # 根据classes文件获取真实目标
            xmin = int(x_centre * width - w * width / 2)  # 坐标转换
            ymin = int(y_centre * height - h * height / 2)
            xmax = int(x_centre * width + w * width / 2)
            ymax = int(y_centre * height + h * height / 2)

            tmp = [name, xmin, ymin, xmax, ymax]  # 单个检测框
            ob.append(tmp)

    # 绘制检测框
    for name, x1, y1, x2, y2 in ob:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # 绘制矩形框
        cv2.putText(image, name, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, color=(0, 0, 255))

        # 保存图像
    if save:
        cv2.imwrite('result.png', image)

        # 展示图像
    cv2.imshow('test', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = '/media/btbu/gt/ljx/FLIR-align-3class/FLIR-align-3class/infrared/val/images/FLIR_09891_PreviewData.jpeg'  # 传入图片


    label_path = '/media/btbu/gt/ljx/multispectral-object-detection-main/runs/test/exp7/labels/FLIR_09891_PreviewData.txt'  # 自动获取相应的txt标签文件

    txtShow(img=img_path, txt=label_path, save=False)


# import cv2
#
#
# def txtShow(img, txt, save=True):
#     image = cv2.imread(img)
#     height, width = image.shape[:2]  # 获取原始图像的高和宽
#
#     # 读取classes类别信息
#     classes = ['Person', 'car', 'bicycle']
#     #
#
#     # 读取yolo格式标注的txt信息
#     with open(txt, 'r') as f:
#         labels = f.read().splitlines()
#     # ['0 0.403646 0.485491 0.103423 0.110863', '1 0.658482 0.425595 0.09375 0.099702', '2 0.482515 0.603795 0.061756 0.045387', '3 0.594122 0.610863 0.063244 0.052083', '4 0.496652 0.387649 0.064732 0.049107']
#
#     ob = []  # 存放目标信息
#     for i in labels:
#         cl, x_centre, y_centre, w, h = i.split(' ')
#
#         # 需要将数据类型转换成数字型
#         cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w), float(h)
#
#         name = classes[cl]  # 根据classes文件获取真实目标
#         xmin = int(x_centre * width - w * width / 2)  # 坐标转换
#         ymin = int(y_centre * height - h * height / 2)
#         xmax = int(x_centre * width + w * width / 2)
#         ymax = int(y_centre * height + h * height / 2)
#
#         tmp = [name, xmin, ymin, xmax, ymax]  # 单个检测框
#         ob.append(tmp)
#
#     # 绘制检测框
#     for name, x1, y1, x2, y2 in ob:
#         cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # 绘制矩形框
#         cv2.putText(image, name, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.5, thickness=1, color=(0, 0, 255))
#
#         # 保存图像
#     if save:
#         cv2.imwrite('result.png', image)
#
#         # 展示图像
#     cv2.imshow('test', image)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     img_path = '/media/btbu/gt/ljx/FLIR-align-3class/FLIR-align-3class/visible/val/images/FLIR_09891_PreviewData.jpg'  # 传入图片
#
#
#     label_path = '/media/btbu/gt/ljx/FLIR-align-3class/FLIR-align-3class/visible/val/labels/FLIR_09891_PreviewData.txt'  # 自动获取相应的txt标签文件
#
#     txtShow(img=img_path, txt=label_path, save=False)




# import cv2
# import os
#
# img_dir = '/media/btbu/gt/ljx/FLIR-align-3class/FLIR-align-3class/infrared/val/images'
# label_dir = '/media/btbu/gt/ljx/multispectral-object-detection-main/runs/test/exp7/labels'
# save_dir = '/media/btbu/gt/ljx/OUT'  # 事先新建一个文件夹，用来存放标注好的图片
#
# lable_file = os.listdir(label_dir)
# img_file = os.listdir(img_dir)
#
# for file in lable_file:
#
#     file_dir = os.path.join(label_dir, file)
#
#     with open(file_dir, 'r') as f:
#         print(os.path.join(img_dir, file.split('.')[0] + '.jpg'))
#         image_src = cv2.imread(os.path.join(img_dir, file.split('.')[0] + '.jpg'))
#         image_row = image_src.shape[0]
#         image_col = image_src.shape[1]
#
#         for line in f.readlines():
#             x_ = float(line.split(' ')[1])
#             y_ = float(line.split(' ')[2])
#             w_ = float(line.split(' ')[3])
#             h_ = float(line.split(' ')[4])
#
#             w = image_col
#             h = image_row
#
#             x1 = w * x_ - 0.5 * w * w_
#             x2 = w * x_ + 0.5 * w * w_
#             y1 = h * y_ - 0.5 * h * h_
#             y2 = h * y_ + 0.5 * h * h_
#
#             draw = cv2.rectangle(image_src, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], 2)
#
#         cv2.imwrite(os.path.join(save_dir, file.split('.')[0] + '.jpg'), draw)
