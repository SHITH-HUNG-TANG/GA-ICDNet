import threading
import os

# 子執行緒類別
class showTensorboard(threading.Thread):
	def __init__(self, dirPath, name, port, showWeb):
		threading.Thread.__init__(self)
		self.dirPath = dirPath
		self.name = name
		self.port = port
		self.showWeb = showWeb
	
	def run(self):
		if self.showWeb:
			import webbrowser
			webbrowser.open("http://localhost:%d/"% (self.port))
		#rootPath = os.path.join(self.dirPath, self.name)
		rootPath = self.dirPath
		print(f"tensorboard --logdir={rootPath} --port={self.port}")
		os.system(f"tensorboard --logdir={rootPath} --port={self.port}")

if __name__ == '__main__':
	# 建立執行緒
	name = "ITCNET-OULP_Bag-[test][3]"
	showTensorboard("./works/", name, 8001).start()
