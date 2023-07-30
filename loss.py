import torch
import torch.nn as nn

# Defining a function to calculate Intersection over Union (IoU)
def iou(box1, box2, is_pred=True):
    if is_pred:
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format
        # import pdb; pdb.set_trace()
        # Box coordinates of prediction
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
  
        # Box coordinates of ground truth
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
  
        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
  
        # Calculate the union area
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection
  
        # Calculate the IoU score
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)
  
        # Return IoU score
        return iou_score
      
    else:
        # IoU score based on width and height of bounding boxes
          
        # Calculate intersection area
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * \
                            torch.min(box1[..., 1], box2[..., 1])
  
        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area
  
        # Calculate IoU score
        iou_score = intersection_area / union_area
  
        # Return IoU score
        return iou_score

class YoloLoss(nn.Module):
	def __init__(self, S=7, B=2, C=5):
		super(YoloLoss, self).__init__()

		self.mse = nn.MSELoss(reduction="sum")
		self.S = S
		self.B = B
		self.C = C
		self.lambda_noobj = 0.5
		self.lambda_coord = 5


	def forward(self, predictions, target):					
		# predictions : bxsxsx15( 5classes, 2anchor boxes) [0:6(class), 5:11(x,y,w,h,obj),11:16(x,y,w,h,ob) ]
		# target : bxsxsx10 (5 class, 1 box) [0:5(classes), 6:11(x,y,w,h,obj)]
		import pdb; pdb.set_trace()
		iou_b1 = iou(predictions[...,6:10], target[...,6:10])
		iou_b2 = iou(predictions[...,11:15], target[...,6:10])

		ious = torch.cat(iou_b1, iou_b2)
		iou_max, best_box = torch.max(ious)

		obj = target[..., 10]

		#localization loss
		box_pred = obj * (
				best_box * (predictions[..., 6:10])
				+ (1 - best_box * (predictions[..., 11:15]))
			)
		box_target = obj * target[..., 6:10]

		box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * torch.sqrt(torch.abs(box_pred[..., 2:4] + 1e-6)) 

		box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

		box_loss = self.mse(
					torch.flatten(box_pred, dim=-2),
					torch.flatten(box_target, dim=-2)
			)

		# Objectness 
		pred_box = best_box * (predictions[..., 10]) + (1 - best_box) * (predictions[..., 15])

		object_loss = self.mse(
						torch.flatten(pred_box * obj),
						torch.flatten(target[...,10] * obj)
			)

		# No object

		# Class loss

		return 1


	# localization loss

	# objectness

	# classification loss



def test():
	loss = YoloLoss()
	pred = torch.randn([1, 7, 7, 15], dtype=torch.float32)
	target = torch.ones([1, 7, 7, 10], dtype=torch.float32)

	out = loss(pred, target)

if __name__ == '__main__':
	test()