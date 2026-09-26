"""Generate the two slip fixtures (needs Pillow). Uses the app's Sarabun font.
    python3 make_slips.py            # writes slip_transfer.jpg, receipt_711.jpg here
Voice fixtures (macOS):  say -v Kanya -o v1.aiff "จ่ายค่ากาแฟ หกสิบห้าบาท" && afconvert -f m4af -d aac v1.aiff v1.m4a
"""
import os
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
from PIL import Image, ImageDraw, ImageFont
F='/Users/ittipong.it/Projects/mint_money/mobile/assets/font/sarabun/Sarabun-'
def f(w,s): return ImageFont.truetype(F+w+'.ttf', s)
# 1) bank transfer slip (payment to a shop)
im=Image.new('RGB',(720,1080),'white'); d=ImageDraw.Draw(im)
d.rectangle([0,0,720,150],fill=(0,160,90)); d.text((40,45),'โอนเงินสำเร็จ',font=f('Bold',52),fill='white')
y=200
for k,v in [('วันที่','24 ก.ย. 2569  12:41'),('จาก','นาย ทดสอบ ระบบ\nธ.กสิกรไทย xxx-x-x1234-x'),('ไปยัง','ร้าน ก๋วยเตี๋ยวเรือป้าแดง\nพร้อมเพย์ xxx-xxx-5678'),('เลขที่รายการ','016267124153AQR01234')]:
    d.text((40,y),k,font=f('Regular',30),fill=(120,120,120)); d.multiline_text((40,y+40),v,font=f('Medium',36),fill='black'); y+=170
d.text((40,y+10),'จำนวนเงิน',font=f('Regular',30),fill=(120,120,120)); d.text((40,y+50),'180.00 บาท',font=f('Bold',64),fill='black')
d.text((40,y+150),'ค่าธรรมเนียม 0.00 บาท',font=f('Regular',30),fill=(120,120,120))
im.save(OUT + 'slip_transfer.jpg',quality=90)
# 2) multi-line store receipt
im=Image.new('RGB',(640,1000),'white'); d=ImageDraw.Draw(im)
d.text((200,30),'7-ELEVEN',font=f('Bold',50),fill='black'); d.text((150,100),'สาขา สีลม คอมเพล็กซ์ (01234)',font=f('Regular',26),fill='black')
d.text((40,150),'25/09/2569 18:22   POS#3  REC#004512',font=f('Regular',24),fill='black')
y=210
for n,a in [('นมสด เมจิ 830มล.','49.00'),('ขนมปังโฮลวีท','35.00'),('น้ำดื่ม 1.5ล. x2','28.00'),('ยาสีฟัน คอลเกต','65.00'),('ข้าวกะเพราไก่ (อุ่น)','45.00')]:
    d.text((40,y),n,font=f('Regular',30),fill='black'); d.text((500,y),a,font=f('Regular',30),fill='black'); y+=55
d.line([40,y+10,600,y+10],fill='black',width=2)
d.text((40,y+30),'รวมทั้งสิ้น',font=f('Bold',36),fill='black'); d.text((470,y+30),'222.00',font=f('Bold',36),fill='black')
d.text((40,y+90),'ชำระโดย TrueMoney Wallet',font=f('Regular',28),fill='black')
im.save(OUT + 'receipt_711.jpg',quality=90)
