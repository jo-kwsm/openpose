import torch
import torch.nn as nn

from libs.models.modules import OpenPose_Feature, make_OpenPose_block


class OpenPoseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.model0 = OpenPose_Feature(pretrained=pretrained)

        self.model1_1 = make_OpenPose_block('block1_1')
        self.model2_1 = make_OpenPose_block('block2_1')
        self.model3_1 = make_OpenPose_block('block3_1')
        self.model4_1 = make_OpenPose_block('block4_1')
        self.model5_1 = make_OpenPose_block('block5_1')
        self.model6_1 = make_OpenPose_block('block6_1')

        self.model1_2 = make_OpenPose_block('block1_2')
        self.model2_2 = make_OpenPose_block('block2_2')
        self.model3_2 = make_OpenPose_block('block3_2')
        self.model4_2 = make_OpenPose_block('block4_2')
        self.model5_2 = make_OpenPose_block('block5_2')
        self.model6_2 = make_OpenPose_block('block6_2')

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)  # PAFs側
        out1_2 = self.model1_2(out1)  # confidence heatmap側

        out2 = torch.cat([out1_1, out1_2, out1], 1)  # 次元1のチャネルで結合
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)

        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)

        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)

        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)

        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        saved_for_loss = []
        saved_for_loss.append(out1_1)  # PAFs側
        saved_for_loss.append(out1_2)  # confidence heatmap側
        saved_for_loss.append(out2_1)
        saved_for_loss.append(out2_2)
        saved_for_loss.append(out3_1)
        saved_for_loss.append(out3_2)
        saved_for_loss.append(out4_1)
        saved_for_loss.append(out4_2)
        saved_for_loss.append(out5_1)
        saved_for_loss.append(out5_2)
        saved_for_loss.append(out6_1)
        saved_for_loss.append(out6_2)

        return (out6_1, out6_2), saved_for_loss


def model_test():
    model = OpenPoseNet()
    model.train()

    batch_size = 2
    dummy_img = torch.rand(batch_size, 3, 368, 368)

    outputs = model(dummy_img)
    print(outputs)


if __name__ == "__main__":
    model_test()
