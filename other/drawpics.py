import matplotlib.pyplot as plt
from sklearn.metrics import auc
from matplotlib.pyplot import MultipleLocator

from utils.dataloader import *



def draw_compare():
    plt.figure()
    # plt.title('', fontsize=20)
    plt.xlabel('positive to negative')
    plt.ylabel('Frame level AUC')

    plt.plot(['1:10', '1:20', '1:30'], [0.72, 0.75, 0.57], label='Meso4', marker='o')
    plt.plot(['1:10', '1:20', '1:30'], [0.60, 0.64, 0.50], label='Xception', marker='s')
    plt.plot(['1:10', '1:20', '1:30'], [0.82, 0.79, 0.57], label='DSP-FWA', marker='^')
    plt.plot(['1:10', '1:20', '1:30'], [0.63, 0.67, 0.56], label='Capsule', marker='*')
    plt.plot(['1:10', '1:20', '1:30'], [0.91, 0.92, 0.78], label='Ours', marker='D')

    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=5)
    plt.savefig('compare.pdf')

def draw_AUC():

    f_fpr = np.load('/home/asus/Code/pvc/m/bs/f_fpr.npy')
    f_tpr = np.load('/home/asus/Code/pvc/m/bs/f_tpr.npy')
    f_roc_auc = auc(f_fpr, f_tpr)
    xcp_f_fpr = np.load('/home/asus/Code/pvc/m/xcp/f_fpr.npy')
    xcp_f_tpr = np.load('/home/asus/Code/pvc/m/xcp/f_tpr.npy')
    xcp_roc_auc = auc(xcp_f_fpr, xcp_f_tpr)
    cap_f_fpr = np.load('/home/asus/Code/pvc/m/cap/f_fpr.npy')
    cap_f_tpr = np.load('/home/asus/Code/pvc/m/cap/f_tpr.npy')
    cap_roc_auc = auc(cap_f_fpr, cap_f_tpr)
    ms4_f_fpr = np.load('/home/asus/Code/pvc/m/ms4/f_fpr.npy')
    ms4_f_tpr = np.load('/home/asus/Code/pvc/m/ms4/f_tpr.npy')
    ms4_roc_auc = auc(ms4_f_fpr, ms4_f_tpr)
    msi_f_fpr = np.load('/home/asus/Code/pvc/m/msi/f_fpr.npy')
    msi_f_tpr = np.load('/home/asus/Code/pvc/m/msi/f_tpr.npy')
    msi_roc_auc = auc(msi_f_fpr, msi_f_tpr)
    plt.figure()
    lw = 2
    plt.plot(f_fpr, f_tpr,
             lw=lw, label='Ours ROC curve (area = %0.2f)' % f_roc_auc)
    plt.plot(xcp_f_fpr, xcp_f_tpr,
             lw=lw, label='Xception ROC curve (area = %0.2f)' % xcp_roc_auc)
    plt.plot(cap_f_fpr, cap_f_tpr,
             lw=lw, label='Capsule ROC curve (area = %0.2f)' % cap_roc_auc)
    plt.plot(ms4_f_fpr, ms4_f_tpr,
             lw=lw, label='Meso4 ROC curve (area = %0.2f)' % ms4_roc_auc)
    plt.plot(msi_f_fpr, msi_f_tpr,
             lw=lw, label='MesoInception4 ROC curve (area = %0.2f)' % msi_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('df_frame.pdf')


def draw_WMW():
    # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    WMW_auc_frame = [0.9032, 0.8772, 0.9383, 0.9293, 0.826, 0.8079]
    WMW_acc_frame = [0.957, 0.957, 0.957, 0.957, 0.957, 0.957]
    WMW_f1_frame = [0.978, 0.978, 0.978, 0.978, 0.978, 0.978]
    WMW_recall_frame = [1, 1, 1, 1, 1, 1]
    WMW_auc_video = [0.908, 0.894, 0.9473, 0.9544, 0.7472, 0.8181]
    WMW_acc_video = [0.957, 0.957, 0.957, 0.957, 0.957, 0.957]
    WMW_f1_video = [0.978, 0.978, 0.978, 0.978, 0.978, 0.978]
    WMW_recall_video = [1, 1, 1, 1, 1, 1]

    x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    plt.figure()
    plt.plot(np.array(x), np.array(WMW_auc_frame), label='AUC')
    # plt.plot(np.array(x), np.array(WMW_auc_video), label='Video level AUC')
    plt.plot(np.array(x), np.array(WMW_acc_frame), label='ACC')
    plt.plot(np.array(x), np.array(WMW_f1_frame), label='F1')
    plt.plot(np.array(x), np.array(WMW_recall_frame), label='recall')
    plt.xlim(0.0, 1)
    plt.legend(loc="lower center")
    plt.xlabel('Margin parameter (gamma) of WMW Loss')
    plt.ylabel('Metrics score')
    plt.savefig('frame_score.pdf')
    plt.show()

    plt.figure()
    plt.plot(np.array(x), np.array(WMW_auc_video), label='AUC')
    # plt.plot(np.array(x), np.array(WMW_auc_video), label='Video level AUC')
    plt.plot(np.array(x), np.array(WMW_acc_video), label='ACC')
    plt.plot(np.array(x), np.array(WMW_f1_video), label='F1')
    plt.plot(np.array(x), np.array(WMW_recall_video), label='recall')
    plt.xlim(0.0, 1)
    plt.legend(loc="lower center")
    plt.xlabel('Margin parameter (gamma) of WMW Loss')
    plt.ylabel('Metrics score')
    plt.savefig('video_score.pdf')
    plt.show()


def draw_auc_compare():
    name = ['Celeb-30', 'Celeb-20', 'Celeb-10']
    our_list = [0.74, 0.95, 0.94]
    w_focal = [0.72, 0.95, 0.91]
    wo_auc = [0.70, 0.90, 0.87]

    x = np.arange(len(name))
    width = 0.25

    plt.bar(x, our_list, width=width, label='Ours')
    plt.bar(x + width, w_focal, width=width, label='Ours with FL', tick_label=name)
    plt.bar(x + 2 * width, wo_auc, width=width, label='Ours with BCE')

    # x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 显示在图形上的值
    # for a, b in zip(x, our_list):
    #     plt.text(a, b + 0.1, b, ha='center', va='bottom')
    # for a, b in zip(x, w_focal):
    #     plt.text(a + width, b + 0.1, b, ha='center', va='bottom')
    # for a, b in zip(x, wo_auc):
    #     plt.text(a + 2 * width, b + 0.1, b, ha='center', va='bottom')

    plt.xticks()
    plt.ylim([0.5, 1.0])
    plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.ylabel('AUC score')
    # plt.xlabel('line')
    # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
    # plt.title("title")
    plt.savefig('w_wo_auc.pdf')
    plt.show()

    # x = list(range(len(our_list)))
    # total_width, n = 0.8, 3
    # width = total_width / n
    #
    # plt.bar(x, our_list, width=width, label='Our')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, w_focal, width=width, label='Our w Focal loss')
    # plt.bar(x, wo_auc, width=width, label='Our w/o AUC loss')
    # plt.xticks(np.array(x) - width / 3, name_list)
    # plt.legend()
    # plt.savefig('w_wo_auc.pdf')
    # plt.show()

    # x = 3
    # total_width, n = 0.8, 3  # 有多少个类型，只需更改n即可
    # width = total_width / n
    # x = x - (total_width - width) / 2
    #
    # plt.bar(x, our_list, width=width, label='Ours')
    # plt.bar(x + width, w_focal, width=width, label='Ours with Focal loss')
    # plt.bar(x + 2 * width, wo_auc, width=width, label='Ours with BCE ')
    #
    # plt.xticks()
    # plt.legend(loc="upper left")  # 防止label和图像重合显示不出来
    # # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.ylabel('AUC score')
    # # plt.xlabel('line')
    # # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # # plt.rcParams['figure.dpi'] = 300  # 分辨率
    # # plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
    # # plt.title("title")
    # plt.savefig('w_wo_auc.pdf')
    # plt.show()


# draw_WMW()
# draw_auc_compare()
# draw_AUC()
draw_compare()