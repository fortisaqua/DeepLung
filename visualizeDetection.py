import matplotlib
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
showid = 0 # from 0 to 4
assert showid in range(5)

anchors = [5, 10, 20]
srcID = '1.3.6.1.4.1.14519.5.2.1.6279.6001.170706757615202213033480003264'
ctdat = np.load('/opt/LUNAPreprocess/subset9/' + srcID + '_clean.npy')
ctlab = np.load('/opt/LUNAPreprocess/subset9/' + srcID +'_label.npy')
pbb = np.load('/opt/DeepLung/detector/results/dpn3d26/retrft960/bbox/' + srcID + '_pbb.npy')
# lbb = np.load('/opt/DeepLung/detector/results/res18/retrft960/val200/' + srcID + '_lbb.npy')

def addRectangle(box, anchor, ax):
    rect = patches.Rectangle((box[2] - anchor, box[1] - anchor), anchor * 2, anchor * 2, linewidth=2, edgecolor='blue',
                             facecolor='none')
    ax.add_patch(rect)

def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)): overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def nms(output, nms_th):
    if len(output) == 0: return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1: bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def showLabel():
    print('Groundtruth')
    print(ctdat.shape, ctlab.shape)
    zs = []
    for idx in xrange(ctlab.shape[0]):
        if abs(ctlab[idx,0])+abs(ctlab[idx,1])+abs(ctlab[idx,2])+abs(ctlab[idx,3])==0: continue
        ax = plt.subplot(1, 1, 1)
        z, x, y = int(ctlab[idx,0]), int(ctlab[idx,1]), int(ctlab[idx,2])
        zs.append(z)
        fig = plt.figure("label_" + str(z))
        lbox = ctlab[idx].astype('int')[:]
        img = np.array(ctdat[0, z, :, :])
        # img[max(0,x-10):min(img.shape[0],x+10), max(0,y-10)] = 255
        # img[max(0,x-10):min(img.shape[0],x+10), min(img.shape[1],y+10)] = 255
        # img[max(0,x-10), max(0,y-10):min(img.shape[1],y+10)] = 255
        # img[min(img.shape[0],x+10), max(0,y-10):min(img.shape[1],y+10)] = 255
        plt.imshow(img, 'gray')
        plt.axis('off')
        rect = patches.Circle((lbox[2], lbox[1]), lbox[3], linewidth=2, edgecolor='red',
                                 facecolor='none')
        ax.add_patch(rect)
        # plt.show()
    return zs

def showPredict(pbb, zs):
    pbb = np.array(pbb[pbb[:,0] > 2])
    pbb = nms(pbb, 0.1)
    print('Detection Results according to confidence')
    print pbb.shape, pbb
    for idx in xrange(pbb.shape[0]):
        z, x, y = int(pbb[idx,1]), int(pbb[idx,2]), int(pbb[idx,3])
        if z in zs:
            fig = plt.figure("predict_" + str(z))
            print z, x, y
            box = pbb[idx].astype('int')[1:]
            imageArray = np.array(ctdat[0, z, :, :])
            ax = plt.subplot(1, 1, 1)
            plt.imshow(imageArray, 'gray')
            plt.axis('off')
            rect = patches.Rectangle((box[2] - box[3], box[1] - box[3]), box[3] * 2, box[3] * 2, linewidth=2, edgecolor='red',
                                     facecolor='none')
            ax.add_patch(rect)
            for anchor in anchors:
                addRectangle(box, anchor, ax)
            # plt.show()


if __name__ == "__main__":
    zs = showLabel()
    showPredict(pbb, zs)
    plt.show()
