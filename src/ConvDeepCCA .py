# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:15:14 2022

@author: mawes
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import ConvDeepCCAModels as models
import time
import logging
import utils

conv_model_types = {"ConvLin": models.ConvLinDeepCCA,
                    "ConvConv": models.ConvConvDeepCCA,
                    "BNBN":models.BNBNDeepCCA,
                    "BNLin":models.BNLinDeepCCA}


class Solver():
    def __init__(self, model, epoch_num, batch_size, learning_rate, reg_par, use_gpu=False):
        self.model = nn.parallel.DistributedDataParallel(model)
        if use_gpu:
            self.model.cuda()
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.use_gpu = use_gpu()

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("CDCCA_Training.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, x1, x2, x1_mask=None, x2_mask=None, vx1=None, vx2=None, vx1_mask=None, vx2_mask=None, tx1=None, tx2=None, tx1_mask=None, tx2_mask=None, checkpoint='checkpoint.model'):
        """
        x1 is the first bottleneck tensors to be used 
        dim=[batch_size, channels, height, width]
        x2 is the second bottleneck tensor to be used.
        dim=[batch_size, features]
        """
        if self.use_gpu:
            x1.cuda()
            x2.cuda()
            if x1_mask is not None:
                x1_mask.cuda()
            if x2_mask is not None:
                x2_mask.cuda()

        data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            if self.use_gpu:
                vx1.cuda()
                vx2.cuda()
                if vx1_mask is not None:
                    vx1_mask.cuda()
                if vx2_mask is not None:
                    vx2_mask.cuda()
        if tx1 is not None and tx2 is not None:
            if self.use_gpu:
                tx1.cuda()
                tx2.cuda()
                if tx2_mask is not None:
                    tx1_mask.cuda()
                if tx2_mask is not None:
                    tx2_mask.cuda()

        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=True))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :, :, :]
                batch_x2 = x2[batch_idx, :]
                if x1_mask is not None:
                    batch_x1_mask = x1_mask[batch_idx, :, :]
                else:
                    batch_x1_mask = x1_mask
                if x2_mask is not None:
                    batch_x2_mask = x2_mask[batch_idx, :, :]
                else:
                    batch_x2_mask = x2_mask
                o1, o2 = self.model(batch_x1, batch_x2,
                                    batch_x1_mask, batch_x2_mask)
                if self.use_gpu:
                    o1.cuda()
                    o2.cuda()
                loss = self.loss(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx1, vx2, vx1_mask, vx2_mask)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint+"_best.pth"))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(),
                                   checkpoint+"_best.pth")
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}, saving model to {}".format(
                            epoch + 1, best_val_loss, checkpoint+"_latest.pth"))
                        torch.save(self.model.state_dict(),
                                   checkpoint+"_latest.pth")
            else:
                torch.save(self.model.state_dict(), checkpoint+"_latest.pth")
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))

        if vx1 is not None and vx2 is not None:
            checkpoint_ = torch.load(checkpoint+"_best.pth")
        else:
            checkpoint_ = torch.load(checkpoint+"_latest.pth")
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss = self.test(vx1, vx2, vx1_mask, vx2_mask)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss = self.test(tx1, tx2, tx1_mask, tx2_mask)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def test(self, x1, x2, x1_mask=None, x2_mask=None):
        with torch.no_grad():
            losses, outputs = self.get_outputs(
                x1, x2, x1_mask, x2_mask)
            return np.mean(losses)

    # def train_linear_cca(self, x1, x2):
        #self.linear_cca.fit(x1, x2, self.outdim_size)

    def get_outputs(self, x1, x2, x1_mask=None, x2_mask=None):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                if x1_mask is not None:
                    batch_x1_mask = x1_mask[batch_idx, :, :]
                else:
                    batch_x1_mask = x1_mask
                if x2_mask is not None:
                    batch_x2_mask = x2_mask[batch_idx, :, :]
                else:
                    batch_x2_mask = x2_mask
                o1, o2 = self.model(batch_x1, batch_x2,
                                    batch_x1_mask, batch_x2_mask)
                if self.use_gpu:
                    o1.cuda()
                    o2.cuda()
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs


def main(args):
    conv_model_class = conv_model_types.get(args['model_type'], None)
    model = conv_model_class(**args)
    if args['pretrained_model'] is not None:
        model.load_state_dict(args['pretrained_model'])
        Solver(model, args['epoch_num'], args['batch_size'],
               args['learning_rate'], args['reg_par'], args['use_gpu'])
    if args['mode'] == "train":
        x1, x2, x1_mask, x2_mask, vx1, vx2, x1_mask, vx2_mask = utils.get_training_data(
            **args)
        Solver.fit(x1, x2, x1_mask, x2_mask, vx1, vx2, x1_mask, vx2_mask, checkpoint=args['checkpoint_name'])
    elif args.mode == "test":
        x1, x2, x1_mask, x2_mask = utils.get_test_data(**args)
        losses, outputs = Solver.get_outputs(x1, x2, x1_mask, x2_mask)
        corr = np.sqrt(-np.mean(losses))
        out = {"corr": corr, "outputs": outputs}
        np.save(args.test_out, out, allow_pickle=True)
if __name__ == '__main__':
    args = utils.get_args(task='segmentation', mode='training')
    main(args)