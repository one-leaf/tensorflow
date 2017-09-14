using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace text2image
{
    class Program
    {
        private static String getFonts()
        {
            String fonts = "阿富汗巴林孟加拉国不丹文莱缅甸柬埔寨塞浦路斯朝鲜香港印度尼西亚伊朗克以色列日本约旦科威特老挝黎嫩澳门马来尔代夫蒙古泊联邦民主共和曼基坦勒菲律宾卡塔沙伯新坡韩里兰叙利泰土耳其酋也越南中台澎金关税区东帝汶哈萨吉库乌兹别洲他家(地)及安哥贝宁博茨瓦纳布隆迪喀麦那群岛佛得角非卜休达乍摩罗刚果提埃赤道几内俄比蓬冈绍肯维毛求洛莫桑米留汪卢旺圣多美普舌昂索撒苏突干法赞津韦托梅士厄立时英德爱意大森堡荷希腊葡萄牙班奥保芬直陀匈冰支敦登挪波力诺瑞典脱陶宛格鲁拜疆白黑山捷伐前顿梵蒂城欧瓜根廷族玻开智伦属圭危海洪都买墨拿秘各丁凯委京皮密陵百慕北斐济盖瑙努图福社会所汤艾帕劳浮洋详合机构际组织性包装原产市辖阳丰石景淀头沟房通州顺义昌平兴怀柔谷云延庆天河红桥丽青辰武清宝坻滨静县蓟省庄长华井陉矿裕藁鹿泉栾正定行唐灵寿高邑深泽皇无极元氏赵辛集晋乐冶润曹妃滦亭迁玉田遵化秦戴抚龙满自治邯郸丛复峰临漳成名涉磁肥乡永年邱鸡广馆魏曲周邢丘柏尧任巨宗宫竞秀莲池苑徐水涞阜容源望易蠡野雄涿碑店张口宣下花园康沽尚蔚万全崇礼承双鹰手营子宽围场沧运光盐肃吴献村回黄骅间廊坊次固厂霸三衡桃枣强饶故冀太小迎杏岭尖草坪娄烦交同郊荣镇浑左盂襄垣屯壶沁潞川朔阴应右仁榆权昔祁遥介湖猗闻喜稷绛夏陆芮忻府五繁峙神岢岚偏汾沃翼洞隰蒲侯霍吕梁离柳楼方孝呼浩赛罕默旗昆仑拐鄂九茂明勃湾松什腾翁牛喇敖汉辽后奈扎郭胜准杭锦审盟赉斡春温陈虎额彦淖磴察卓资商凉四王锡二连嘎珠穆仆寺镶蓝善沈姑铁于岗甘旅鞍千岫岩溪桓振凤凌站鲅鱼圈边细彰宏伟弓灯盘洼银调兵建票葫芦绥绿农树惠潭船蛟桦舒磐梨公江辉靖宇乾扶余洮们珲外依木志常齐锋碾讷冠恒滴麻鹤向工萝鸭贤友谊让胡肇杜岔好翠峦带星上嘉荫佳进风远七茄牡棱逊孙奎玛漠汇闸虹杨闵奉玄淮邺鼓栖霞雨六溧淳塘宜贾铜沛睢沂邳钟坛相熟仓如启皋赣灌涟盱眙响射扬邗仪征邮徒句姜堰宿豫沭泗浙拱墅萧桐庐曙鄞象姚慈瓯苍浔柯虞诸暨嵊婺衢游舟岱椒环仙居缙遂畲徽瑶蜀巢芜镜弋鸠为蚌埠禹庵谢八潘当涂含烈濉官狮观枞潜岳歙黟滁琅琊谯颍界首埇砀璧亳涡贵至郎泾绩旌尾闽厦思翔莆厢涵荔屿流尤将鲤芗霄诏政邵夷汀蕉屏柘鼎谱萍湘栗修彭渝分月章贡信犹寻峡袁载樟铅横鄱历槐崂李胶即淄薛峄儿滕垦烟芝罘牟招潍寒朐兖微祥邹乳照莒钢郯费聊莘茌沾棣菏单郓鄄郑管街巩荥封符杞许尉考瀍涧嵩汝偃师顶卫湛叶郏舞殷滑壁淇浚牧获焦作解放陟濮范鄢葛漯郾召陕渑卧淅邓浉始潢息项驻驿蔡舆确泌级划岸硚陂十茅箭郧竹伍点军猇秭归枝樊荆掇刀感悟梦监滋团浠蕲穴咸随曾恩施苗架芙蓉心麓浏株淞攸茶炎醴韶晖雁蒸耒步君汨澧植益赫沅郴桂禾零冷滩牌溆晃侗芷底泸凰丈番禺从增浈圳斗汕濠潮澄禅坎廉雷电端要紫壮莞揭榕郁良邕鸣融叠彩恭梧圩藤岑防钦覃绵业贺昭峨仫佬等凭琼棠涯崖儋指迈重涪渡坝碚綦足黔潼垫忠节巫柱酉羊堂郫邛崃沿攀蔺邡梓羌油剑阁犍研夹沐彝眉充部陇阆雅珙筠邻蓥渠经棉简藏理壤若孜孚炉稻拖觉冕烽真仡务湄习毕雍碧阡晴贞谟册亨秉穗匀瓮独呈禄劝麒麟傣冲巧蒗洱祜佤澜耿楚谋个旧弥砚畴版勐漾濞巍颇芒盈怒傈僳堆则迦结仲聂类隅乃囊措查错浪申戈札噶革改勤灞未央阎户耀渭岐彬旬功起勉略脂柞峪积祝掖崆峒酒煌岷宕两迭碌湟互助循晏久杂称谦令峻嘴吾磨坂碱吐鄯坤奇垒精音楞轮犁且末焉耆硕车恰疏附莎伽策敏蕴位可境先人币盾铢镑桶闭镀锌铝圆板纤塑料琵琶罐箱漏再生纸袋/席编薄膜硬璃瓷条筐膨箩笼物铺材散裸挂捆然座辆艘套只件把块卷副片份幅对棵筒盆具疋担扇盒亿伏升尺吨短司斤磅盎码寸毫制批打匹发枚粒瓶舱净种样标每品总航蹲领域企实验室素模页证50pm12934678PQ.RSNMVWL";
            return fonts;
        }

        private static Bitmap text2bitmap(string text,string savefile)
        {
            Font font = new Font("宋体", 9, FontStyle.Regular);
            Size sz = TextRenderer.MeasureText(text, font);
            Bitmap bitmap = new Bitmap(sz.Width, sz.Height, PixelFormat.Format24bppRgb);
            Graphics obj = Graphics.FromImage(bitmap);
            obj.Clear(Color.White);
            obj.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.Default;
            obj.TextRenderingHint = System.Drawing.Text.TextRenderingHint.SystemDefault;
            obj.DrawString(text, font, Brushes.Black, 0, 0);
            bitmap.Save("D:/tensorflow/5/7/data/"+savefile+".png", ImageFormat.Png);
            return bitmap;
        }

        static void Main(string[] args)
        {
            String fonts = getFonts();
            // 做10万张样本用于学习
            List<string> valueList = new List<string>();
            Random rnd = new Random();
            for (int i = 0; i < 100000; i++)
            {
                int fontsLen = rnd.Next(5, 20);
                string value = string.Empty;
                while (value.Length < fontsLen)
                {
                    value += fonts[rnd.Next(fonts.Length)].ToString();
                }                
                text2bitmap(value,i.ToString());
                valueList.Add(i.ToString() + " " + value);
            }
            System.IO.File.WriteAllLines("D:/tensorflow/5/7/data/index.txt", valueList);
        }
    }
}
