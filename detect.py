from utils.dataloaders import IMG_FORMATS, VID_FORMATS
import torch
from models.common import DetectMultiBackend
import argparse
from utils.general import ( cv2, non_max_suppression,  scale_boxes,  xyxy2xywh)
from utils.plots import Annotator, save_one_box
from utils.dataloaders import  LoadImages,letterbox
from distant import object_point_world_position
img_path=''
import numpy as np

# in_mat=np.array( [[4.76257371e+03, 0.00000000e+00, 2.41574317e+03],
#  [0.00000000e+00, 4.73920861e+03, 1.02210531e+03],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


in_mat=np.array( [[2.38257371e+03, 0.00000000e+00, 1.2574317e+03],
 [0.00000000e+00, 2.36920861e+03, 0.52210531e+03],
 [0.00000000e+00, 0.00000000e+00, 1.0000000e+00]])
#自己
# in_mat=np.array([[2545.5,0,863.9843],
#               [0,2528.5,437.1350],
#             [0,0,1]])



out_mat=np.array([[0.00E+00,	1.00E+00,	0.00E+00,	0.00],
[0.00E+00,	0.00E+00,	1.00E+00,	0.00E+00],
[1.00E+00,	0.00E+00,	0.00E+00,	0.00E+00],
[0.00E+00,	0.00E+00,	0.00E+00,	1.00E+00]])

#k=np.array([0,0.,0,0.11,0.22])
#k=np.array([0,0.,0,0,0])
#张
k=[ 0.04083242 , 0.50256544 ,-0.00699 ,    0.00383466 ,-1.72877006]
#,,左下角往后折,
import PIL.Image as Image
save_dir ='./results'
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time
def run(weights= r'',#权重来源
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        source=r'E:\allcode\easy_yolo\car',#测试集来源
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        ):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(weights, device=device)


    model.eval()

    stride, names, pt = model.stride, model.names, model.pt

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride,in_mat=in_mat,k=k)
    # Process predictions
    for path, im, im0s, vid_cap, s in dataset:

        im = torch.from_numpy(im).to(model.device)

        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        seen, windows = 0, [],
        # Inference

        visualize = False

        pred = model(im, augment=augment, visualize=visualize)

        # NMS

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        """计数"""



        for i, det in enumerate(pred):  # per image
            seen += 1

            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir +'/'+ p.name)  # im.jpg
            txt_path = str(save_dir +'/'+ 'labels' +'/'+ p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            "ours"
            people_num=0
            car_num=0


            if len(det):
                # Rescale boxes from img_size to im0 size

                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                      # Add bbox to image
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    c = int(cls)  # integer class
                    if c==0 :
                        people_num+=1
                        print(xywh)
                    if c==2 or c==5:
                        car_num+=1

                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label)#改变颜色

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            im0 = annotator.result()
            cv2.imwrite(save_path, im0)
            print('the people_num is'+str(people_num))
            print('the car_num is' + str(car_num))

def run_video(weights= r'',#权重来源
        video_path='',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,
        max_det=1000,
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,
        vid_stride=1,
        auto=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(weights, device=device)

    stride, names, pt = model.stride, model.names, model.pt
    capture = cv2.VideoCapture(video_path)
    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGRtoRGB
        frame = cv2.flip(frame, -1)
        frame=cv2.undistort(frame, np.array(in_mat), np.array(k))
        im0 = frame.copy()

        h_or=im0.shape[0]
        w_or=im0.shape[1]


        im = letterbox(frame, imgsz, stride=stride, auto=auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(model.device)

        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        seen, windows = 0, [],
        # Inference

        visualize = False

        pred = model(im, augment=augment, visualize=visualize)

        # NMS

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        "ours"
        people_num = 0
        car_num = 0

        det=pred[0]

        if len(det):
            # Rescale boxes from img_size to im0 size

            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class


            # Write results

            for *xyxy, conf, cls in reversed(det):

                # Add bbox to image
                c = int(cls)  # integer class
                if c not in [0,1,2,3,5,7]:
                    continue
                if c == 0 :
                    people_num += 1
                if c == 2 or c == 5 or c == 1or c == 7 or c == 3:
                    car_num += 1
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                dis=object_point_world_position(xywh[0]*w_or,xywh[1]*h_or,xywh[2]*w_or,xywh[3]*h_or,p=out_mat,k=in_mat)

                #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                label=names[c]+' '+f'{dis:.2f}'
                annotator.box_label(xyxy, label)  # 改变颜色

        im0 = annotator.result()

        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("result", 720, 1280)
        cv2.imshow("result", im0)
        cv2.waitKey(2)


    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=r'E:\allcode\easy_yolo\yolov5s.pt', help='./output/weight')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    #run(**vars(opt))
    run_video(weights=r'E:\allcode\easy_yolo\yolov5s.pt',video_path='F:/gyt.mp4')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)