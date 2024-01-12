from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt # pip install matplotlib
import numpy as np  # pip install numpy
from numpy import interp
from sklearn.preprocessing import label_binarize
import pandas as pd # pip install pandas

target_loc = "query-large.txt"     # 真实标签所在的文件
target_data = pd.read_csv(target_loc, sep="\t", names=["loc","type"])
true_label = [i for i in target_data["type"]]
margin = 0.375
b = 0.25
models = [
    (margin, "a"),
    (margin+b, "b"),
    (margin+2*b, "c"),
    (margin+3*b, "d"),
    (margin+4*b, "e"),
    (margin+5*b, "f"),
    (margin+6*b, "g"),
    #(margin+7*b, "h"),
    #("AMAM-Net", "i")
]

for net_name,number in models:
    predict_loc = r"D:/PyTorch/OSL-pt/csv/similarity/margin/margin-{}.csv".format(net_name)

    predict_data = pd.read_csv(predict_loc, header=0)
    data = predict_data.to_numpy()
    predict_label = np.argmax(data, axis=1)
    predict_score = np.max(data, axis=1)
    print(len(predict_label))
    print(len(predict_score))

    # 精度，准确率， 预测正确的占所有样本种的比例
    accuracy = accuracy_score(true_label, predict_label)
    print("精度: ", accuracy)

    # 查准率P（准确率），precision(查准率)=TP/(TP+FP)

    precision = precision_score(true_label, predict_label, labels=None, pos_label=1,
                                average='macro')  # 'micro', 'macro', 'weighted'
    print("查准率P: ", precision)

    # 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
    recall = recall_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
    print("召回率: ", recall)

    # F1-Score
    f1 = f1_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
    print("F1 Score: ", f1)

    fontsize_title = 17
    family = 'Times New Roman'

    from matplotlib.colors import LinearSegmentedColormap

    label_names = ["Eng.", "HiS.", "Dev.", "PoS.", "Sai.", "Sur.", "Tra.","Pas.", "Und."]
    confusion = confusion_matrix(true_label, predict_label, labels=[i for i in range(len(label_names))])

    cmap = plt.cm.Blues
    # plt.figure(figsize=(8, 8))
    plt.matshow(confusion, cmap=cmap)  # Greens, Blues, Oranges, Reds

    for i in range(len(confusion)):
        row_sum = sum(confusion[i])  # 计算每一行的总和
        for j in range(len(confusion)):
            percentage = confusion[j, i] / row_sum * 100  # 计算百分比
            # percentage = round(percentage, 1)  # 四舍五入保留一位小数
            plt.annotate(f'{percentage:.1f}%', xy=(i, j), horizontalalignment='center', verticalalignment='center',
                         family=family, fontsize=11.5)

    plt.ylabel('True label', family=family, fontsize=13)
    plt.xlabel('Predicted label', family=family, fontsize=13)
    plt.xticks(range(len(label_names)), label_names, family=family, fontsize=11.5)
    plt.yticks(range(len(label_names)), label_names, family=family, fontsize=11.5)
    plt.title("({}) Vector-Dimension-{}".format(number, net_name), family=family, fontsize=14.5)
    fig = plt.gcf()
    fig.set_size_inches(4.65, 4)
    #plt.savefig('./fig/optimizer/pdf/{}-cm.pdf'.format(net_name), format='pdf')
    #plt.savefig('./fig/optimizer/png/{}-cm.png'.format(net_name), format='png', dpi=600)
    #plt.show()

    n_classes = len(label_names)
    # binarize_predict = label_binarize(predict_label, classes=[i for i in range(n_classes)])
    binarize_predict = label_binarize(true_label, classes=[i for i in range(n_classes)])

    # 读取预测结果

    predict_score = predict_data.to_numpy()

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binarize_predict[:, i], [socre_i[i] for socre_i in predict_score])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    # Convert RGB values from 0-255 to 0-1 range
    colors = [(r / 255, g / 255, b / 255) for r, g, b in
              [(2, 48, 71), (14, 91, 118), (26, 134, 163),
               (70, 172, 202), (155, 207, 232), (243, 249, 252),
               (255, 202, 95), (254, 168, 9), (251, 132, 2)]]

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print(fpr["macro"])
    print(tpr["macro"])
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='black', linestyle='-.', linewidth=2)

    df = pd.DataFrame({'fpr': fpr["macro"], 'tpr': tpr["macro"]})
    # 将DataFrame保存到csv文件中
    df.to_csv('./csv/aroc/average_roc-{}.csv'.format(net_name), index=False)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.2f})'.format(label_names[i], roc_auc[i]),
                 color=colors[i])

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, family=family)
    plt.ylabel('True Positive Rate', fontsize=14, family=family)
    plt.title("({}) Margin-{}".format(number, net_name),fontsize=18,family=family)
    plt.legend(loc="best", fontsize=12, prop={'family': 'Times New Roman'})
    plt.savefig('./fig/margin/pdf/{}-roc.pdf'.format(net_name), format='pdf')
    plt.savefig('./fig/margin/png/{}-roc.png'.format(net_name), format='png',dpi=600)
    #plt.show()

    n_classes = len(label_names)
    # binarize_predict = label_binarize(predict_label, classes=[i for i in range(n_classes)])
    Y_test = label_binarize(true_label, classes=[i for i in range(n_classes)])
    y_score = predict_data.to_numpy()

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    # print("recall:",recall)
    # print(average_precision)

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

    # print("precision:",precision)
    print(average_precision)

    import pandas as pd

    # 绘制平均PR曲线
    display = PrecisionRecallDisplay(recall=recall["micro"], precision=precision["micro"],
                                     average_precision=average_precision["micro"])
    df = pd.DataFrame({'recall': recall["micro"], 'precision': precision["micro"]})
    # 将DataFrame保存到csv文件中
    df.to_csv('./csv/apr/average_pr_{}.csv'.format(net_name), index=False)
    display.plot()
    _ = display.ax_.set_xlabel("Recall", fontdict={'family': 'Times New Roman', 'size': 12})
    _ = display.ax_.set_ylabel("Precision", fontdict={'family': 'Times New Roman', 'size': 12})
    _ = display.ax_.legend(prop={'family': 'Times New Roman', 'size': 17})
    _ = display.ax_.set_title("({}) Margin-{}".format(number, net_name), fontdict={'family': 'Times New Roman', 'size': 19})
    plt.savefig('./fig/margin/pdf/{}-apr.pdf'.format(net_name), format='pdf')
    plt.savefig('./fig/margin/png/{}-apr.png'.format(net_name), format='png', dpi=600)
    #plt.show()
    print('save finished')

    # 绘制每个类的PR曲线和 iso-f1 曲线
    # setup plot details
    colors = [(r / 255, g / 255, b / 255) for r, g, b in
              [(2, 48, 71), (14, 91, 118), (26, 134, 163),
               (70, 172, 202), (155, 207, 232), (243, 249, 252),
               (255, 202, 95), (254, 168, 9), (251, 132, 2)]]

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", linestyle=':', alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.01), family=family)

    display = PrecisionRecallDisplay(recall=recall["micro"], precision=precision["micro"],
                                     average_precision=average_precision["micro"])

    display.plot(ax=ax, name="Micro-average precision-recall", linestyle=':', color="black", lw=2)

    label_names = ["Engineering", "High Speed", "Ocean Development", "Port Service", "Sailing", "Surface", "Transport",
                   "Tugs", "Underwater"]

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i],
                                         average_precision=average_precision[i], )
        display.plot(ax=ax, name="Precision-recall for {0}".format(label_names[i]), color=colors[i], lw=2)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])

    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, family=family)
    plt.ylabel('True Positive Rate', fontsize=14, family=family)
    ax.legend(handles=handles, labels=labels, loc="best", fontsize=12, prop={'family': 'Times New Roman'})
    ax.set_title("embedding=0.5-Extension of Precision-Recall curve to multi-class", fontsize=17, family=family)
    #plt.show()