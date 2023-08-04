import torch
import torch.nn as nn
from utils.util import iou


class YoloLoss(nn.Module):
	def __init__(self, num_classes, S=7, B=2, C=5, ):
		super(YoloLoss, self).__init__()

		self.mse = nn.MSELoss(reduction="sum")
		self.S = S
		self.B = B
		self.C = C
		self.lambda_noobj = 0.5
		self.lambda_coord = 5
		self.num_classes = num_classes


	def forward(self, predictions, target):					
		# predictions : bxsxsx15( 5classes, 2anchor boxes) [0:6(class), 5:11(x,y,w,h,obj),11:16(x,y,w,h,ob) ]
		# target : bxsxsx10 (5 class, 1 box) [0:5(classes), 6:11(x,y,w,h,obj)]

		
		iou_b1 = iou(predictions[...,5:9], target[...,5:9])
		iou_b2 = iou(predictions[...,10:14], target[...,5:9])

		ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
		iou_max, best_box = torch.max(ious, dim=0)

		best_box = best_box.unsqueeze(3)

		iobj = target[..., 9].unsqueeze(3)
		
		#localization loss
		box_pred = iobj * (
				best_box * predictions[..., 5:9]
				+ (1 - best_box) * (predictions[..., 10:14])
			)
		box_target = iobj * target[..., 5:9]

		box_pred[..., 1:3] = torch.sign(box_pred[..., 1:3]) * torch.sqrt(torch.abs(box_pred[..., 1:3] + 1e-6)) 

		box_target[..., 1:3] = torch.sqrt(box_target[..., 1:3])

		box_loss = self.mse(
					torch.flatten(box_pred, end_dim=-2),
					torch.flatten(box_target, end_dim=-2)
			)
		
		# Object loss 
		pred_box = (best_box * predictions[..., 9]) + ((1 - best_box) * predictions[..., 14])

		object_loss = self.mse(
						torch.flatten(pred_box * iobj),
						torch.flatten(target[...,9] * iobj)
			)

		# No object loss

		no_object_loss = self.mse(
						torch.flatten( (1-iobj) * predictions[...,9], start_dim=1),
						torch.flatten( (1 -iobj) * target[..., 9], start_dim=1)
		)

		no_object_loss += self.mse(
						torch.flatten( (1-iobj) * predictions[...,14], start_dim=1),
						torch.flatten( (1 -iobj) * target[..., 9], start_dim=1)
		)

		# Class loss

		class_loss = self.mse(
						torch.flatten( iobj * predictions[...,:5], start_dim=-2),
						torch.flatten( iobj * target[..., :5], start_dim=-2)
		)

		loss = (
				self.lambda_coord * box_loss
				+ object_loss
				+ self.lambda_noobj * no_object_loss
				+ class_loss
		)
		# print(f"box_loss = {box_loss}")
		# print(f"object_loss = {object_loss}")
		# print(f"no_object_loss = {no_object_loss}")
		# print(f"class_loss = {class_loss}")
		return loss

class Yolov3Loss(nn.Module):
	def __init__(self, num_classes, num_anchors=3):
		super().__init__()

		self.mse = nn.MSELoss(reduction="sum")
		self.bce = nn.BCEWithLogitsLoss(reduction="sum")
		self.cross_entropy = nn.CrossEntropyLoss()
		self.sigmoid = nn.Sigmoid()

		self.num_classes = num_classes
		self.num_anchors = num_anchors
	
		self.lambda_noobj = 10
		self.lambda_coord = 10
		self.lambda_obj = 1
		self.lambda_class = 1


	def forward(self, pred, target, anchor):
		import pdb; pdb.set_trace()
		obj = target[..., 0] == 1
		noobj = target[..., 0] == 0

		# No obj loss

		no_object_loss = self.bce(
			(pred[..., 0:1][noobj]), (target[..., 0:1][noobj])
		)

		# object loss

		anchor = anchor.reshape(1, 3, 1, 1, 2)
		box_pred = torch.cat([self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5])* anchor], dim=-1)
		ious = iou(box_pred[obj], target[..., 1:5][obj]).detach()
		object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj])

		# localization loss (box)
		pred[..., 1:3] = self.sigmoid(pred[..., 1:3]) # x, y
		target[..., 3:5] = torch.log(
			(1e-6 + target[..., 3:5] / anchor)
		) # w, h
		box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])

		# class loss
		class_loss = self.cross_entropy(
			(pred[..., 5:][obj], (target[..., 5][obj].long()))
		)

		loss = (self.lambda_coord * box_loss
				+ self.lambda_obj + object_loss
				+ self.lambda_noobj + no_object_loss
				+ self.lambda_class + class_loss
		)
		return loss

def test():
	target = torch.tensor([1, 0, 0, 0, 0, 2.5, 2, 3, 2, 1])
	pred = torch.tensor([0.5, 0.2, 0.1, 0.2, 0.0, 2.5, 3, 3, 2, 0.2, 3.25, 2, 1.5, 2, 0.8])
	predv = pred.view(1, 1, 1, 15)
	targetv = target.view(1, 1, 1, 10)
	prede = predv.expand(1, 7, 7, 15)
	targete = targetv.expand(1, 7, 7, 10)
	loss = YoloLoss(5)
	# pred = torch.randn([1, 7, 7, 15], dtype=torch.float32)
	# target = torch.randn([1, 7, 7, 10], dtype=torch.float32)

	out = loss(prede, targete)








if __name__ == '__main__':
	test()