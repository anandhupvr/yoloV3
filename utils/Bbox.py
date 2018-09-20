class Bbox:
	def __init__(self, x_ax, w_ax,y_ax, h_ax, cat=None):
		self.x_ax = x_ax
		self.y_ax = y_ax
		self.w_ax = w_ax
		self.h_ax = h_ax
		self.cat = cat

	def __getitem__(self, key):
		return getattr(self, key)