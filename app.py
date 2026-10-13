"""AirCanvas Studio — webcam-controlled virtual drawing board.

Python 3.10–3.12 recommended. Model is downloaded automatically on first launch.
Press H for gesture instructions. Gesture recognition is heuristic rather than a trained
classifier; camera angle, lighting and hand pose affect reliability.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
OUTPUT = ROOT / 'outputs'
TOOLS = ['Pencil', 'Pen', 'Marker', 'Highlighter', 'Brush', 'Eraser']
COLORS = [(32, 40, 234), (235, 130, 28), (55, 178, 44), (183, 72, 170), (22, 22, 22), (255, 255, 255)]  # BGR
COLOR_NAMES = ['Coral', 'Blue', 'Green', 'Purple', 'Black', 'White']
BG = (247, 245, 239)
INK = (46, 45, 43)


def get_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print('Downloading the official MediaPipe hand landmark model (first run only)...')
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if MODEL_PATH.stat().st_size < 100_000:
            raise RuntimeError('Model download appears incomplete')
        return MODEL_PATH
    except Exception as exc:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError('Could not download hand_landmarker.task. Connect to the internet and run again, '
                           f'or download {MODEL_URL} and save it at {MODEL_PATH}. Error: {exc}') from exc


def label(frame, message, pos, scale=.55, color=INK, thickness=1):
    cv2.putText(frame, message, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def rr(frame, x1, y1, x2, y2, color, radius=14, fill=True, thickness=1):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    if x2 <= x1 or y2 <= y1:
        return
    radius = max(1, min(int(radius), (x2-x1)//2, (y2-y1)//2))
    t = -1 if fill else thickness
    cv2.rectangle(frame, (x1+radius,y1), (x2-radius,y2), color, t)
    cv2.rectangle(frame, (x1,y1+radius), (x2,y2-radius), color, t)
    for cx,cy,start,end in ((x1+radius,y1+radius,180,270),(x2-radius,y1+radius,270,360),
                            (x2-radius,y2-radius,0,90),(x1+radius,y2-radius,90,180)):
        cv2.ellipse(frame,(cx,cy),(radius,radius),0,start,end,color,t,cv2.LINE_AA)


def blend_card(frame, x1,y1,x2,y2, color=(255,255,255), alpha=.94, radius=17):
    roi = frame[y1:y2,x1:x2]
    if roi.size == 0: return
    over = np.zeros_like(roi)
    rr(over,0,0,x2-x1-1,y2-y1-1,color,radius)
    mask=np.any(over!=0,axis=2)
    roi[mask]=(roi[mask].astype(np.float32)*(1-alpha)+over[mask].astype(np.float32)*alpha).astype(np.uint8)


def inside(xy, rect):
    return rect[0] <= xy[0] < rect[2] and rect[1] <= xy[1] < rect[3]


def angle_distance(a,b):
    return float(math.hypot(a[0]-b[0], a[1]-b[1]))


def gesture(landmarks, w, h):
    """Classify straightforward finger extension; normalized against hand size."""
    pts=np.array([(p.x*w,p.y*h) for p in landmarks],dtype=np.float32)
    # Straightness is measured as tip-to-MCP over MCP-to-PIP and tip-to-PIP.
    extended=[]
    for tip,pip,mcp in ((8,6,5),(12,10,9),(16,14,13),(20,18,17)):
        straight=angle_distance(pts[tip],pts[mcp]) / max(1.,angle_distance(pts[pip],pts[mcp]))
        extended.append(straight>1.73 and angle_distance(pts[tip],pts[0])>angle_distance(pts[pip],pts[0])*1.08)
    # Thumb excluded from fist check: thumb may rest across palm.
    fist=not any(extended)
    draw=extended[0] and not any(extended[1:])
    peace=extended[0] and extended[1] and not any(extended[2:])
    return tuple(map(int,pts[8])),fist,draw,peace,pts

@dataclass
class Stroke:
    tool: str
    color: tuple[int,int,int]
    size: int
    points: list[tuple[int,int]]


class AirCanvas:
    def __init__(self,w=1280,h=720):
        self.w,self.h=w,h
        self.tool='Pen'; self.color_index=4; self.size=7
        self.strokes:list[Stroke]=[];self.redo:list[Stroke]=[];self.active:Stroke|None=None
        self.paper=np.full((h,w,3),255,dtype=np.uint8)
        self.needs_render=True
        self.palette=False; self.show_camera=False;self.show_help=False
        self.fist_start=None; self.fist_armed=True;self.fist_was=False
        self.hover=None;self.hover_at=None
        self.pointer=None; self.smoothed=None
        self.last_draw=False;self.toast='Open palm to rest | Fist to choose tools';self.toast_until=time.monotonic()+5
        self.last_frame=None

    def notify(self,message):
        self.toast=message;self.toast_until=time.monotonic()+2.4

    def start_stroke(self,pt):
        self.active=Stroke(self.tool,COLORS[self.color_index],self.size,[pt])
        self.needs_render=True

    def finish_stroke(self):
        if self.active:
            self.strokes.append(self.active)
            self.redo.clear();self.active=None
            self.needs_render=True

    def undo(self):
        self.finish_stroke()
        if self.strokes:
            self.redo.append(self.strokes.pop());self.needs_render=True;self.notify('Undo')

    def redo_action(self):
        if self.redo:
            self.strokes.append(self.redo.pop());self.needs_render=True;self.notify('Redo')

    def clear(self):
        self.finish_stroke();self.strokes.clear();self.redo.clear();self.needs_render=True;self.notify('Canvas cleared')

    def save(self):
        self.finish_stroke(); self.render_paper()
        OUTPUT.mkdir(parents=True,exist_ok=True)
        path=OUTPUT/f'aircanvas_{datetime.now():%Y%m%d_%H%M%S}.png'
        ok=cv2.imwrite(str(path),self.paper)
        self.notify(f'Saved: {path.name}' if ok else 'Save failed')
        if ok: print(f'Saved drawing: {path}')

    def choose(self,tool):
        self.finish_stroke();self.tool=tool;self.palette=False;self.hover=None
        self.notify(f'{tool} selected')

    def draw_stroke(self, base, stroke):
        pts=stroke.points
        if not pts:return
        tool=stroke.tool
        size=max(1,stroke.size)
        if tool=='Pencil': size=max(1,round(size*.48))
        elif tool=='Marker':size=max(5,round(size*2.6))
        elif tool=='Highlighter':size=max(9,round(size*3.4))
        elif tool=='Brush':size=max(3,round(size*1.6))
        elif tool=='Eraser':size=max(13,round(size*3.0))
        color=(255,255,255) if tool=='Eraser' else stroke.color
        # Render each semi-transparent stroke onto a separate overlay to avoid
        # cumulatively darkening overlaps of consecutive segments.
        overlay=base.copy() if tool in ('Marker','Highlighter','Pencil') else base
        if len(pts)==1:
            cv2.circle(overlay,pts[0],max(1,size//2),color,-1,cv2.LINE_AA)
        for i in range(1,len(pts)):
            width=size
            if tool=='Brush':
                speed=angle_distance(pts[i],pts[i-1]);width=max(2,int(size*(1.15-min(speed,42)/65)))
            cv2.line(overlay,pts[i-1],pts[i],color,width,cv2.LINE_AA)
            cv2.circle(overlay,pts[i],max(1,width//2),color,-1,cv2.LINE_AA)
        alpha={'Marker':.54,'Highlighter':.25,'Pencil':.80}.get(tool,1.)
        if alpha<1:cv2.addWeighted(overlay,alpha,base,1-alpha,0,dst=base)

    def render_paper(self):
        if not self.needs_render:return
        self.paper[:]=(255,255,255)
        for s in self.strokes:self.draw_stroke(self.paper,s)
        if self.active:self.draw_stroke(self.paper,self.active)
        self.needs_render=False

    def palette_rects(self):
        ww= min(980,self.w-40); x=(self.w-ww)//2; gap=8
        cell=(ww-34-gap*5)//6
        return [(x+17+i*(cell+gap),self.h//2-40,x+17+i*(cell+gap)+cell,self.h//2+95) for i in range(6)]

    def update(self,landmarks,now):
        if landmarks is None:
            self.finish_stroke();self.pointer=None;self.smoothed=None
            self.fist_start=None;self.fist_armed=True;self.fist_was=False
            self.hover=None;return
        tip,fist,draw,peace,pts=gesture(landmarks,self.w,self.h)
        if self.smoothed is None:self.smoothed=np.array(tip,dtype=float)
        else:self.smoothed=.47*np.array(tip)+.53*self.smoothed
        pointer=tuple(map(int,self.smoothed));self.pointer=pointer
        if fist:
            self.finish_stroke()
            if self.fist_start is None:self.fist_start=now
            if self.fist_armed and (now-self.fist_start)>.52:
                self.palette=not self.palette;self.fist_armed=False
                self.hover=None;self.hover_at=None
                self.notify('Tool selector open' if self.palette else 'Tool selector closed')
        else:
            self.fist_start=None;self.fist_armed=True
        if self.palette:
            self.finish_stroke()
            idx=next((i for i,rect in enumerate(self.palette_rects()) if inside(pointer,rect)),None) if not fist else None
            if idx!=self.hover:self.hover=idx;self.hover_at=now
            if idx is not None and self.hover_at is not None and now-self.hover_at>.72:
                self.choose(TOOLS[idx])
            return
        # Top toolbar is also mouse-clickable. Drawing only happens below it.
        valid=draw and not fist and pointer[1]>110
        if valid:
            p=(max(0,min(self.w-1,pointer[0])),max(0,min(self.h-1,pointer[1])))
            if self.active is None:self.start_stroke(p)
            elif angle_distance(self.active.points[-1],p)>1.8:
                self.active.points.append(p);self.needs_render=True
        else:self.finish_stroke()

    def keypress(self,key):
        if key in (ord('q'),27):return False
        if key==ord('s'):self.save()
        elif key==ord('z'):self.undo()
        elif key==ord('y'):self.redo_action()
        elif key==ord('c'):self.clear()
        elif key==ord('h'):self.show_help=not self.show_help
        elif key==ord('v'):self.show_camera=not self.show_camera
        elif key==ord('t'):self.palette=not self.palette;self.finish_stroke()
        elif key==ord('['):self.size=max(2,self.size-2);self.notify(f'Brush size: {self.size}')
        elif key==ord(']'):self.size=min(35,self.size+2);self.notify(f'Brush size: {self.size}')
        elif ord('1')<=key<=ord('6'):
            self.color_index=key-ord('1');self.notify(f'Color: {COLOR_NAMES[self.color_index]}')
        return True

    def on_mouse(self,event,x,y,flags,userdata):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        if self.palette:
            for i,r in enumerate(self.palette_rects()):
                if inside((x,y),r):self.choose(TOOLS[i]);return
        if y<101:
            for i in range(6):
                if inside((x,y),(290+i*41,43,326+i*41,79)):
                    self.color_index=i;self.notify(f'Color: {COLOR_NAMES[i]}');return
            if inside((x,y),(self.w-280,41,self.w-189,83)):self.undo();return
            if inside((x,y),(self.w-184,41,self.w-94,83)):self.save();return
            if inside((x,y),(self.w-90,41,self.w-9,83)):self.clear();return

    def draw_ui(self,frame,mode):
        w,h=self.w,self.h
        blend_card(frame,12,12,w-12,101,alpha=.94,radius=18)
        label(frame,'AIR / CANVAS',(29,45),.78,INK,2)
        label(frame,f'{self.tool.upper()}   {self.size}px',(29,77),.48,(115,112,107),1)
        for i,color in enumerate(COLORS):
            cx=308+i*41;cy=61
            if i==self.color_index:cv2.circle(frame,(cx,cy),20,(75,70,64),2,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),14,color,-1,cv2.LINE_AA)
            if i==5:cv2.circle(frame,(cx,cy),14,(170,170,170),1,cv2.LINE_AA)
        for text_,rect in [('UNDO',(w-280,41,w-189,83)),('SAVE',(w-184,41,w-94,83)),('CLEAR',(w-90,41,w-9,83))]:
            rr(frame,*rect,(236,233,227),10)
            label(frame,text_,(rect[0]+13,rect[1]+27),.52,INK,1)
        if self.palette:self.draw_palette(frame)
        if self.pointer is not None:
            color=(19,180,104) if mode=='DRAW' and not self.palette else (43,138,225)
            cv2.circle(frame,self.pointer,12,color,2,cv2.LINE_AA)
            cv2.circle(frame,self.pointer,3,color,-1,cv2.LINE_AA)
        blend_card(frame,12,h-55,w-12,h-12,alpha=.92,radius=12)
        label(frame,f'{mode}  |  FIST: TOOLS   INDEX: DRAW   1-6: COLOR   [ ]: SIZE   Z/Y: UNDO/REDO   S: SAVE   H: HELP   Q: QUIT',
              (27,h-28),.45,(72,69,66),1)
        if time.monotonic()<self.toast_until:
            tw=min(w-100,max(320,len(self.toast)*11))
            x=(w-tw)//2
            blend_card(frame,x,111,x+tw,151,(35,38,43),.90,12)
            label(frame,self.toast[:78],(x+15,137),.55,(255,255,255),1)
        if self.show_help:self.help_ui(frame)

    def draw_palette(self,frame):
        x1,y1,x2,y2=20,self.h//2-122,self.w-20,self.h//2+154
        blend_card(frame,x1,y1,x2,y2,(248,247,244),.97,22)
        label(frame,'CHOOSE A DRAWING TOOL',(x1+26,y1+44),.82,INK,2)
        label(frame,'Point at a tile and hold for 0.7s   |   Close fist again to dismiss',(x1+27,y1+73),.52,(98,96,93))
        for i,r in enumerate(self.palette_rects()):
            active=self.tool==TOOLS[i];hover=self.hover==i
            rr(frame,*r,(207,227,238) if hover else ((221,230,221) if active else (231,229,223)),13)
            col=(40,40,40)
            cy=r[1]+42;cx=(r[0]+r[2])//2
            size=[2,5,14,22,10,18][i]
            cv2.line(frame,(cx-22,cy+12),(cx+22,cy-12),(125,165,210) if i==3 else col,size,cv2.LINE_AA)
            label(frame,TOOLS[i].upper(),(r[0]+7,r[3]-18),.43,INK,1)
            if hover and self.hover_at:
                progress=min(1.,(time.monotonic()-self.hover_at)/.72)
                cv2.rectangle(frame,(r[0]+7,r[3]-8),(r[0]+7+int((r[2]-r[0]-14)*progress),r[3]-4),(30,163,104),-1)

    def help_ui(self,frame):
        x1,y1,x2,y2= max(80,self.w//2-360),self.h//2-155,min(self.w-80,self.w//2+360),self.h//2+170
        blend_card(frame,x1,y1,x2,y2,(248,248,248),.98,18)
        lines=['HAND GESTURES','Index finger only   =   draw','Open hand / two fingers   =   stop drawing',
               'Closed fist (hold 0.5s)   =   open/close tool picker',
               'Point at tool (hold 0.7s)   =   choose tool','KEYBOARD','1-6 color   [ / ] size   Z undo   Y redo',
               'S save PNG   C clear   V camera background','T tool picker   H close help   Q / Esc quit']
        for i,s in enumerate(lines):label(frame,s,(x1+24,y1+34+i*31),.63 if i in (0,5) else .54,INK,2 if i in (0,5) else 1)

    def frame(self,webcam=None,mode='READY'):
        self.render_paper();frame=self.paper.copy()
        if self.show_camera and webcam is not None:
            frame=cv2.addWeighted(webcam,.45,frame,.55,0)
        self.draw_ui(frame,mode)
        return frame


def main():
    parser=argparse.ArgumentParser(description='AI virtual drawing board with fist-triggered brush menu')
    parser.add_argument('--camera',type=int,default=0,help='Webcam index, default 0')
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    args=parser.parse_args()
    if args.width<900 or args.height<600:parser.error('For this UI, choose at least 900x600')
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise SystemExit('Missing packages. Run: python -m pip install -r requirements.txt') from exc
    model=get_model()
    cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW if sys.platform=='win32' else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    if not cap.isOpened():raise SystemExit('Cannot open webcam. Check Windows camera permissions, close Zoom/Teams, or try --camera 1.')
    app=AirCanvas(args.width,args.height)
    options=vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,num_hands=1,min_hand_detection_confidence=.55,
        min_hand_presence_confidence=.55,min_tracking_confidence=.50)
    win='AirCanvas Studio  |  Q to quit'
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win,args.width,args.height)
    cv2.setMouseCallback(win,app.on_mouse)
    t0=time.monotonic();last_timestamp=-1
    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                good,image=cap.read()
                if not good:
                    print('Webcam frame not available; stopping.');break
                image=cv2.flip(image,1)
                image=cv2.resize(image,(app.w,app.h))
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                timestamp=max(last_timestamp+1,int((time.monotonic()-t0)*1000));last_timestamp=timestamp
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),timestamp)
                landmarks=result.hand_landmarks[0] if result.hand_landmarks else None
                now=time.monotonic()
                app.update(landmarks,now)
                mode='NO HAND' if landmarks is None else 'TOOL MENU' if app.palette else ('DRAW' if app.active else 'HAND READY')
                cv2.imshow(win,app.frame(image,mode))
                key=cv2.waitKey(1)&0xFF
                if key!=255 and not app.keypress(key):break
                if cv2.getWindowProperty(win,cv2.WND_PROP_VISIBLE)<1:break
    except KeyboardInterrupt:pass
    finally:
        app.finish_stroke();cap.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
