import torch
import sklearn.metrics as metrics

class mAP:
    """mean average precision: 1/|Right| * sum( P@k )
    """
    def __init__(self):
        self.type = 0

    def eval_scalar(self,pred_s, true_s):
        if pred_s.shape[1] > 1 or true_s.shape[1] > 1:
            print('Inputs need to be a torch scalar!')

    def eval_vector(self,pred_mat, true_mat, bins = None ):
        """mean average precision with input of vectored labels:
            pred_mat: N*C matrix
            true_mat: N*C matrix
        """
        if not (torch.is_tensor(pred_mat) and torch.is_tensor(true_mat)):
            print('Inputs need to be a torch tensor!')

        num_classes = pred_mat.shape[1]
        num_samples = pred_mat.shape[0]
        if bins is None:
            K = num_samples
        else:
            K = bins
        pred_sorted, idx_mat = torch.sort(pred_mat,dim=0, descending=True)
        precisions = torch.zeros(num_classes)
        for i in range(num_classes):
            idx = idx_mat[:,i]
            # x = true_mat[idx,i]
            x = torch.index_select(true_mat[:,i],0,idx)
            y = torch.cumsum(x,dim=0)
            num = torch.FloatTensor(range(num_samples))+1
            y /= num
            precisions[i] = torch.mean(y[:K])

        map = torch.mean(precisions)

        return map

    def eval_matrix(self,pred_mat, true_mat, bins = None ):
        """mean average precision with input of vectored labels:
            pred_mat: N*C_1 matrix
            true_mat: N*C_2 matrix
        """
        if not (torch.is_tensor(pred_mat) and torch.is_tensor(true_mat)):
            print('Inputs need to be a torch tensor!')

        num_bins = pred_mat.shape[1]
        num_samples = pred_mat.shape[0]
        num_classes = true_mat.shape[1]
        if bins is None:
            K = num_samples
        else:
            K = bins

        # calculating similarity matrix
        pred_s = pred_mat.mm(pred_mat.t())
        pred_s = torch.div(pred_s,torch.diag(pred_s))
        true_s = true_mat.mm(true_mat.t())
        idx_rm = [i for i, v in enumerate(torch.diag(true_s)) if v == 0]
        # np.savetxt(os.path.join('data', 'tmp.csv'), torch.diag(true_s).numpy(), fmt="%d")
        true_s = torch.div(true_s, torch.diag(true_s))
        pred_sorted, idx_mat = torch.sort(pred_s,dim=0, descending=True)

        precisions = torch.zeros(num_samples)
        for i in set(range(num_samples))-set(idx_rm):
            idx = idx_mat[:,i]
            x = torch.index_select(true_s[:,i],0,idx)
            y = torch.cumsum(x,dim=0)
            num = torch.FloatTensor(range(num_samples))+1
            y /= num
            precisions[i] = torch.mean(y[:K])

        map = torch.mean(precisions)

        return map

class accuracy:
    """accuracy:
    """
    def __init__(self):
        self.type = 0

    def eval(self,pred_v, true_v):
        # calulate the weighted accuracy or unbalanced accuracy
        idx_a = [i for i, value in enumerate(pred_v) if pred_v[i] == true_v[i]]
        acc_weighted = float(len(idx_a))/float(len(pred_v))
        # calculate the unweighted accuracy or balanced accuracy
        labels = torch.unique(true_v)
        acc = torch.zeros(len(labels))
        for i in range(len(labels)):
            idx_c = [j for j in range(len(true_v)) if true_v[j] == labels[i]]
            acc[i] = torch.sum(pred_v[idx_c] == true_v[idx_c]).double()/float(len(idx_c))
        acc_unweighted = torch.mean(acc)
        return acc_weighted, acc_unweighted

class f1score:
    """f1score: weighted and unweighted
    """

    def __init__(self):
        self.type = 0

    def eval(self, pred_v, true_v):
        # calulate the weighted f1 score
        f1 = metrics.f1_score(true_v.float(), pred_v.float(), average='micro')
        f1_weighted = metrics.f1_score(true_v.float(), pred_v.float(), average='macro')
        return f1, f1_weighted