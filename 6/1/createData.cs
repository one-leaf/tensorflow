using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.IO;

namespace text2image
{
    class Program
    {
        private static String getFonts()
        {
            String fonts = "";
            for (int i=32; i <= 126; i++)
            {
                fonts = fonts + Char.ConvertFromUtf32(i);
            }
            
            for  (int i=int.Parse("4E00",System.Globalization.NumberStyles.HexNumber); i<=int.Parse("9FBB",System.Globalization.NumberStyles.HexNumber); i++)
            {
                fonts = fonts + Char.ConvertFromUtf32(i);
            }
            fonts = fonts + "。？！，、；：「」『』‘’“”（）〔〕【】—…–．《》〈〉";
            return fonts;
        }

        private static Bitmap text2bitmap(string text,string savefile)
        {
            float fontsize = 9F;
            Font font = new Font("宋体", fontsize, FontStyle.Regular);
            Size sz = TextRenderer.MeasureText(text, font);
            Bitmap bitmap = new Bitmap(sz.Width, sz.Height, PixelFormat.Format24bppRgb);
            Graphics obj = Graphics.FromImage(bitmap);
            obj.Clear(Color.White);
            obj.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.Default;
            obj.TextRenderingHint = System.Drawing.Text.TextRenderingHint.SystemDefault;
            obj.DrawString(text, font, Brushes.Black, 0, 0);
            bitmap.Save(savefile, ImageFormat.Png);
            return bitmap;
        }

        static void createFonts(String outDir)
        {
            String fonts = getFonts();
            // 做200万张样本用于学习
            List<string> valueList = new List<string>();
            Random rnd = new Random();
            for (int j = 0; j < 10; j++)
            {
                String savePath = outDir + j.ToString() ;
                if (!Directory.Exists(savePath))
                {
                    Directory.CreateDirectory(savePath);    
                }
                for (int i = 0; i < 200000; i++)
                {
                    int fontsLen = rnd.Next(4, 20);
                    string value = string.Empty;
                    while (value.Length < fontsLen)
                    {
                        value += fonts[rnd.Next(fonts.Length)].ToString();
                    }
                    string saveName = j.ToString() + "/" + i.ToString()+".png";
                    text2bitmap(value, outDir + saveName);
                    valueList.Add(saveName + " " + value);
                }
            }
            System.IO.File.WriteAllLines(outDir + "index.txt", valueList);
        }

        static void Main(string[] args)
        {
            createFonts("D:/ocr/data/");
           // text2bitmap("1234567890", "d:/0.png");
           // text2bitmap("ABCDEFeaefefef", "d:/1.png");
          //  text2bitmap("(1.2/34567890)", "d:/2.png");
           // text2bitmap("中国", "d:/3.png");
        }
    }
}
