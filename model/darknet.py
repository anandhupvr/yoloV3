import torch
import torch.nn as nn


class ConvBlock(nn.Module):
	def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
		super(ConvBlock, self).__init__()

		self.conv =  nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding, bias=False)
		self.norm = nn.BatchNorm2d(output_channel)
		self.act = nn.LeakyReLU()

	def forward(self, x):
		x = self.conv(x)
		x = self.norm(x)
		x = self.act(x)
		return x

class ResidualBlock(nn.Module):
	def __init__(self, input_channel):
		super(ResidualBlock, self).__init__()

		reduced_channel = int(input_channel/2)

		self.conv_batch1 = ConvBlock(input_channel, reduced_channel, kernel_size=1, padding=0)
		self.conv_batch2 = ConvBlock(reduced_channel, input_channel)

	def forward(self, x):

		residual = x
		# import pdb; pdb.set_trace()
		x = self.conv_batch1(x)
		x = self.conv_batch2(x)

		x += residual

		return x

class Darknet53(nn.Module):
	def __init__(self):
		super(Darknet53, self).__init__()

		# self.num_classes = num_classes

		self.conv1 = ConvBlock(3, 32)
		self.conv2 = ConvBlock(32, 64, stride=2)
		self.residual_block1 = self.make_blocks(64, 1)
		self.conv3 = ConvBlock(64, 128, stride=2)
		self.residual_block2 = self.make_blocks(128, 2)
		self.conv4 = ConvBlock(128, 256, stride=2)
		self.residual_block3 = self.make_blocks(256, 8)
		self.conv5 = ConvBlock(256, 512, stride=2)
		self.residual_block4 = self.make_blocks(512, 8)
		self.conv6 = ConvBlock(512, 1024, stride=2)
		self.residual_block5 = self.make_blocks(1024, 4)


	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.residual_block1(x)
		x = self.conv3(x)
		x = self.residual_block2(x)
		x = self.conv4(x)
		x = self.residual_block3(x)
		route_1 = x
		x = self.conv5(x)
		x = self.residual_block4(x)
		route_2 = x
		x = self.conv6(x)
		route_3 = self.residual_block5(x)

		return route_1, route_2, route_3




	def make_blocks(self, input_channel, num_blocks):
		layers = []
		for i in range(num_blocks):
			layers.append(ResidualBlock(input_channel))

		return nn.Sequential(*layers)

# class YoloDetection(nn.Module):
# 	def __init__(self, input_channel, output_channel):
# 		super(YoloDetection, self).__init__()

# 		self.conv1 = ConvBlock(1024, 512, kernel_size=1)
# 		self.conv2 = ConvBlock(512, 1024, kernel_size=3)
# 		self.conv3 = ConvBlock(1024, 512, kernel_size=1)
# 		self.conv4 = ConvBlock(512, 1024, kernel_size=3)
# 		self.conv5 = ConvBlock(1024, 512, kernel_size=1)
# 		self.conv_large = ConvBlock(512, 1024, kernel_size=3)
# 		self.conv_large_bbox =  nn.Conv2d(1024, output_channel,  1, bias=False)

# 	def forward(self, x):
# 		x = 


class YoloV3(nn.Module):
	def __init__(self, num_classes):
		super(YoloV3, self).__init__()

		self.num_classes = num_classes
		output_channel = 3*(self.num_classes + 5)
		#route_1 : 256, 52, 52	  first layer
		# route_2 : 512, 26, 26   middle layer
		# route_3 : 1024, 13, 13  last layer	

		# route_1, route_2, route_3 = Darknet53()
		self.darknet53 = Darknet53()
		# route_3 large
		# # 3*(self.num_classes + 5) num_anchors * (num_classes + 5 (x,y,w, h), objectness?)
		self.convl1 = ConvBlock(1024, 512, kernel_size=1, padding=0)
		self.convl2 = ConvBlock(512, 1024, kernel_size=3)
		self.convl3 = ConvBlock(1024, 512, kernel_size=1, padding=0)
		self.convl4 = ConvBlock(512, 1024, kernel_size=3)
		self.convl5 = ConvBlock(1024, 512, kernel_size=1, padding=0)
		self.convl_obj = ConvBlock(512, 1024, kernel_size=3)
		self.convl_bbox =  nn.Conv2d(1024, output_channel,  1, bias=False)

		# for route_2
		self.convl_1x1 = ConvBlock(512, 256, kernel_size=1, padding=0) # convl5
		#upsample
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

		# self.concat = torch.cat([self.up1, route_2], dim=-1)
		self.convm1 = ConvBlock(768, 256, kernel_size=1, padding=0)
		self.convm2 = ConvBlock(256, 512, kernel_size=3)
		self.convm3 = ConvBlock(512, 256, kernel_size=1, padding=0)
		self.convm4 = ConvBlock(256, 512, kernel_size=3)
		self.convm5 = ConvBlock(512, 256, kernel_size=1, padding=0)
		self.convm_obj = ConvBlock(256, 512, kernel_size=3)
		self.convm_bbox = nn.Conv2d(512, output_channel, 1, bias=False)

		self.convm_1x1 = ConvBlock(256, 128, kernel_size=1, padding=0) # convm5

		# route_1
		# self.concat2 = torch.cat([self.up2, route_1])
		self.convs1 = ConvBlock(384, 128, kernel_size=1, padding=0)
		self.convs2 = ConvBlock(128, 256, kernel_size=3)
		self.convs3 = ConvBlock(256, 128, kernel_size=1, padding=0)
		self.convs4 = ConvBlock(128, 256, kernel_size=3)
		self.convs5 = ConvBlock(256, 128, kernel_size=1, padding=0)
		self.convs_obj = ConvBlock(128, 256, kernel_size=3)
		self.convs_bbox = nn.Conv2d(256, output_channel, 1, bias=False)		


	def forward(self, x):

		feat = self.darknet53(x)
		x1 = self.convl1(feat[2])
		x2 = self.convl2(x1)
		x3 = self.convl3(x2)
		x4 = self.convl4(x3)
		x5 = self.convl5(x4)
		lobj = self.convl_obj(x5)
		lbbox = self.convl_bbox(lobj)

		x6 = self.convl_1x1(x5)
		x7 = self.up(x6)
	
		x = torch.cat([x7, feat[1]], dim=1)
		x = self.convm1(x)
		x = self.convm2(x)
		x = self.convm3(x)
		x = self.convm4(x)
		x = self.convm5(x)
		mobj = self.convm_obj(x)
		mbbox = self.convm_bbox(mobj)

		x = self.convm_1x1(x)
		x = self.up(x)

		x = torch.cat([x, feat[0]], dim=1)
		x = self.convs1(x)
		x = self.convs2(x)
		x = self.convs3(x)
		x = self.convs4(x)
		x = self.convs5(x)
		sobj = self.convs_obj(x)
		sbbox = self.convs_bbox(sobj)		

		lbbox_scaled = lbbox.reshape(
			lbbox.shape[0], 3, self.num_classes + 5, lbbox.shape[2], lbbox.shape[3]
			).permute(0, 1, 3, 4, 2)

		mbbox_scaled = mbbox.reshape(
			mbbox.shape[0], 3, self.num_classes + 5, mbbox.shape[2], mbbox.shape[3]
			).permute(0, 1, 3, 4, 2)

		sbbox_scaled = sbbox.reshape(
			sbbox.shape[0], 3, self.num_classes + 5, sbbox.shape[2], sbbox.shape[3]
			).permute(0, 1, 3, 4, 2)
		
		return lbbox_scaled, mbbox_scaled, sbbox_scaled


def test():
	x = torch.randn((1, 3, 416, 416))
	model = Darknet53()
	out1, out2, out3 = model(x)
	print(f"out1 : {out1.shape}")
	print(f"out2 : {out2.shape}")
	print(f"out3 : {out3.shape}")

	print(" ----------------")

	yolov3 = YoloV3(5)
	lbbox, mbbox, sbbox = yolov3(x)

	print(f"lbbox : {lbbox.shape}")
	print(f"mbbox : {mbbox.shape}")
	print(f"sbbox : {sbbox.shape}")



if __name__ == "__main__":
	test()