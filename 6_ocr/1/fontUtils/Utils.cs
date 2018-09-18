using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;

namespace RenderFontHttpServer
{
    class Utils
    {
      public static String[] engFontNames = new String[]{
            "Arial","Calibri","Cambria","Candara","Comic Sans MS","Consolas",
            "Constantia","Corbel","Courier New","Courier","Fixedsys",
            "Franklin Gothic","Gabriola","Georgia","Impact","Lucida Console",
            "Lucida Sans Unicode","Microsoft Sans Serif","Palatino linotype",
            "MS Sans Serif","MS Serif","Nina","Segoe","Segoe UI",
            "Small Fonts","System","Tahoma","Terminal","Times New Roman",
            "Trebuchet MS","Verdana"
        };
        public static String[] chiFontNames = new String[] {
            "方正兰亭超细黑简体","方正舒体","方正姚体",
            "仿宋","黑体","华文仿宋","华文行楷","华文楷体","华文隶书","华文宋体",
            "华文细黑","华文新魏","华文中宋","楷体","隶书","宋体","微软雅黑",
            "新宋体","幼圆"
        };

        public static IEnumerable<String> GetEngFonts()
        {
            return engFontNames;
        }
        public static IEnumerable<String> GetChiFonts()
        {
            return chiFontNames;
        }

        public static Bitmap Render(string text, string fontname, int fontsize, int fontmode, int renderhint)
        {
            Bitmap bmp = new Bitmap(1, 1);
            Graphics graphics = Graphics.FromImage(bmp);
            FontStyle style;
            switch (fontmode)
            {
                case 1:
                    style = FontStyle.Bold;
                    break;
                case 2:
                    style = FontStyle.Italic;
                    break;
                case 4:
                    style = FontStyle.Underline;
                    break;
                case 8:
                    style = FontStyle.Strikeout;
                    break;
                default:
                    style = FontStyle.Regular;
                    break;
            }
            Console.WriteLine("{0} {1} {2} {3}", fontname,fontsize,fontmode,text);
            Font font = new Font(fontname, fontsize, style);
            SizeF stringSize = graphics.MeasureString(text, font);
            bmp = new Bitmap(bmp, (int)stringSize.Width, (int)stringSize.Height);
            graphics = Graphics.FromImage(bmp);
            graphics.Clear(Color.White);
            graphics.CompositingQuality = CompositingQuality.HighQuality;
            graphics.InterpolationMode = InterpolationMode.HighQualityBilinear;
            graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
            graphics.SmoothingMode = SmoothingMode.HighQuality;

            switch (renderhint)
            {
                case 1:
                    graphics.TextRenderingHint = TextRenderingHint.SingleBitPerPixelGridFit;
                    break;
                case 2:
                    graphics.TextRenderingHint = TextRenderingHint.SingleBitPerPixel;
                    break;
                case 3:
                    graphics.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;
                    break;
                case 4:
                    graphics.TextRenderingHint = TextRenderingHint.AntiAlias;
                    break;
                case 5:
                    graphics.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;
                    break;
                default:
                    graphics.TextRenderingHint = TextRenderingHint.SystemDefault;
                    break;
            }
            graphics.DrawString(text, font, Brushes.Black, 0, 0);
            font.Dispose();
            graphics.Flush();
            graphics.Dispose();
            return bmp;
        }
    }
}
