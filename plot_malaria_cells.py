## for plotting
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import matplotlib.pyplot as plt

## for loading history
import json

def plot_loss_acc(history_file, epochs):
    with open(history_file, 'r') as fp:
        train_val_history = json.load(fp)

    train_loss = train_val_history['loss']
    val_loss = train_val_history['val_loss']
    train_acc = train_val_history['acc']
    val_acc = train_val_history['val_acc']

    ## plot training/validation loss
    plt.figure()
    plt.plot(range(epochs), train_loss, 'r-', label="Training Loss")
    plt.plot(range(epochs), val_loss, 'b-', label="Validation Loss")
    plt.legend()
    plt.ylabel("Cross Entropy Loss")
    plt.xlabel("Epochs")
    plt.title("Training and Validation Loss")
    plt.savefig("train_val_loss.pdf", bbox_inches='tight')

    ## plot training/validition accuracy
    plt.figure()
    plt.plot(range(epochs), train_acc, 'r--', label="Training Acc")
    plt.plot(range(epochs), val_acc, 'b--', label="Validation Acc")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.title("Training and Validation Accuracy")
    plt.savefig("train_val_acc.pdf", bbox_inches='tight')


def plot_curves(y_true, y_pred):
    ## get FPR/TPR for AUROC
    fpr, tpr, threshold = roc_curve(y_true, y_pred)

    ## plot AUROC curve
    plt.figure()
    plt.plot(fpr, tpr, 'b-', label="AUC=%0.2f"%auc(fpr, tpr))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC")
    plt.savefig("test_AUROC.pdf", bbox_inches='tight')

    ## Get Precision/Recall for AUPR
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    ## plot AUPR curve
    plt.figure()
    plt.plot(recall, precision, 'b-', label="F1=%0.2f"%f1_score(y_true, y_pred))
    plt.legend(loc = 'lower left')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title("Precision/Recall Curve")
    plt.savefig("test_AUPR.pdf", bbox_inches='tight')


## plot out AUROC
# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#
## Compute micro-average ROC curve and ROC area
#fpr["malaria"], tpr["malaria"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#plt.figure()
#lw = 2
#plt.plot(fpr[2], tpr[2], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()


