import os
import numpy as np
import math
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, roc_auc_score, auc

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import operator


def save_confusion_matrix(y_target, y_pred, save_dir):
    labels = ['Angrily disgusted', 'Angrily surprised', 'Angry', 'Appalled', 'Awed', 'Disgusted', 'Fearful', 'Fearfully angry', 'Fearfully surprised', 'Happily disgusted', 'Happily surprised', 'Happy', 'Sad', 'Sadly angry', 'Sadly disgusted', 	'Surprised']
    cm = confusion_matrix(y_target, y_pred, np.arange(16))
    plot_confusion_matrix(cm, labels, normalize=False)
    '''fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    #ax.set_xticklabels([''] + labels, fontsize=6)
    #ax.set_yticklabels([''] + labels, fontsize=6)
    ax.xaxis.set_ticklabels(labels, fontsize=6)
    ax.yaxis.set_ticklabels(labels, fontsize=6)
    plt.xlabel('Predicted')
    plt.ylabel('True')'''
    plt.savefig(os.path.join(save_dir, 'Confusion_Matrix.png'))


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()

def plot_roc_curves(y_target, y_predict, cls, save_dir):
    labels = ['Angrily disgusted', 'Angrily surprised', 'Angry', 'Appalled', 'Awed', 'Disgusted', 'Fearful', 'Fearfully angry', 'Fearfully surprised', 'Happily disgusted', 'Happily surprised', 'Happy', 'Sad', 'Sadly angry', 'Sadly disgusted', 	'Surprised']
    y_target = label_binarize(y_target, classes=cls)
    y_predict = label_binarize(y_predict, classes=cls)
    n_classes = y_target.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_target[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_target.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure(figsize=(12, 10))
    lw = 2
    #plt.plot(fpr[2], tpr[2], color='darkorange',
    #         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    for i, label in zip(range(n_classes), labels):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(label, roc_auc[i]))
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'ROC_Curves.png'))   # save the figure to file



def get_accuracy_per_epoch(classifier, models_dir, model_label, save_dir):
    acc = []
    if os.path.exists(models_dir):
        load_epoch = 1
        for file in os.listdir(models_dir):
            if file.startswith(model_label + "_epoch_"):
                #load_epoch = max(load_epoch, int(os.path.splitext(file)[0].split('_')[2]))
                classifier._opt.load_epoch = load_epoch
                load_epoch = load_epoch + 1
                a = classifier.eval(load_last_epoch = False)
                acc.append(a)
        plt.figure(figsize=(12, 10))
        lw = 2
        plt.plot(np.arange(len(acc)), np.array(acc), lw=lw, label='Accuracy')

        plt.xlabel('Accuracy')
        plt.ylabel('Epoch Number')
        plt.title('Accuracy with the number of epochs')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, 'Accuracy.png'))   #

    else:
        assert os.path.exists(models_dir), 'Folder for models: ( %s ) not found' % models_dir


def geometricMean(arr, n) :

    # declare product variable and
    # initialize it to 1.
    product = 1

    # Compute the product of all the
    # elements in the array.
    for i in range(0,n) :
        product = product * arr[i]

    # compute geometric mean through
    # formula pow(product, 1/n) and
    # return the value to main function.
    gm = (float)(math.pow(product, (1 / float(n))))
    return (float)(gm)

def create_aus_lookup():

    labels = {'angrily_disgusted':0, 'angrily_surprised':1, 'angry':2, \
    'appalled':3, 'awed':4, 'disgusted':5, \
    'fearful':6, 'fearfully_angry':7, 'fearfully_surprised':8, \
    'happily_disgusted':9, 'happily_surprised':10, 'happy':11, \
    'sad':12, 'sadly_angry':13, 'sadly_disgusted':14, 	'surprised':15}

    #au_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '12', '14', '15', '17', '20', '23', '25', '26', '45']

    aus_dict = dict()
    aus_dict['happy'] = [8,14]
    aus_dict['sad'] = [2,10]
    aus_dict['fearful'] = [0,2,12,14]
    aus_dict['angry'] = [2,5,-1] # 24 is not encoded in action units??
    aus_dict['surprised'] = [0,1,14,15]
    aus_dict['disgusted'] = [6, 7, 11]
    aus_dict['happily_surprised'] = [0,1,8,14]
    aus_dict['happily_disgusted'] = [7,8,14]
    aus_dict['sadly_disgusted'] = [2,7]
    aus_dict['fearfully_angry'] = [2,12,14]
    aus_dict['fearfully_surprised'] = [0,1,3,12,14]
    aus_dict['sadly_angry'] = [2,5,10]
    aus_dict['angrily_surprised'] = [2,14,15]
    aus_dict['appalled'] = [2,6,7]
    aus_dict['angrily_disgusted'] = [2,7,11]
    aus_dict['awed'] = [0,1,3,14]

    return aus_dict, labels


def create_sample(aus_dict, labels):
    cond = np.random.random_sample((17,))

    conf = dict()
    for k,v in aus_dict.items():
        conf[k] = geometricMean(cond[v], len(v))

    name = max(conf.iteritems(), key=operator.itemgetter(1))[0]
    label = labels[name]

    sample = {'data':cond, 'label':label}
    return sample, name
