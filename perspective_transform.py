import tkinter as tk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter.filedialog

def order_points(pts):
	rect = np.zeros((4,2), dtype="float32")
	# top-left top-right bottom-right bottom-left

	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]		#top-left
	rect[2] = pts[np.argmax(s)] 	#bottom-right

	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]	#top-right
	rect[3] = pts[np.argmax(diff)]	#bottom-left

	return rect

def perspective_transformation(img, pts):
	# get ordered points and unpack them
	rect = order_points(pts)
	(tl, tr, br, bl)=rect

	# compute width of new image
	# distance between topmost two points
	widthT = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
	# distance between bottommost two points
	widthB = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)

	# compute height
	#distance between leftmost points
	heightL = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
	# distance between rightmost points
	heightR = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)

	# find max height and width
	maxWidth = max(int(widthB), int(widthT))
	maxHeight = max(int(heightL), int(heightR))

	print('width:'+str(maxWidth))
	print('height:'+str(maxHeight))

	dst = np.float32([[0,0],
		[maxWidth-1,0],
		[maxWidth-1, maxHeight-1],
		[0,maxHeight-1]])

	# compute perspective transform matrix
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

	return warped

def draw(event, x, y, flags, param):
	global btn_down
	if event == cv2.EVENT_LBUTTONDBLCLK and len(pts)<4: 
		btn_down = True
		cv2.circle(image,(x,y), 5, (255,0,0), -1)
		if len(pts)>0:
			cv2.line(image, pts[-1], (x, y), (0,0,0), 2)
		cv2.imshow('Draw',image)
		param=(x,y)
		pts.append(param)
		
	elif len(pts)>=4:
		cv2.line(image, pts[-1], pts[0], (0,0,0), 2)
		#cv2.putText(image,'Press any key to exit', (5,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1,cv2.LINE_AA )
		cv2.imshow('Draw',image)
	
	return

root = tk.Tk()
File = tkinter.filedialog.askopenfilename(parent=root, 
                                          initialdir="C:/",
                                          title='Choose an image.')
img = Image.open(File)
image = np.array(img)

#image = cv2.imread('draw.jpg')
cv2.namedWindow('Draw')
cv2.imshow('Draw',image)
pts=list()
pts.append(cv2.setMouseCallback('Draw', draw))
del pts[0]

cv2.waitKey(0)
cv2.line(image, pts[-1], pts[0], (0,0,0), 1)
cv2.imshow('Draw',image)
cv2.destroyAllWindows()

print (pts)

pts = np.float32(pts)
warped = perspective_transformation(image,pts)


h,w,c = warped.shape
# create canvas that can fit the image
canvas = tk.Canvas(root, width=w, height=h)
canvas.pack()
# use PIL to convert Numpy ndarray to a PhotoImage
# add a photo image to the canvas
photo = ImageTk.PhotoImage(image=Image.fromarray(warped))
canvas.create_image(0,0,image=photo, anchor=tk.NW)
root.mainloop()

plt.subplot(121),plt.imshow(image),plt.title('Input')
plt.subplot(122),plt.imshow(warped),plt.title('Output')
plt.show()
