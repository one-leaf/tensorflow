using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Web;

namespace RenderFontHttpServer
{
    public class Request
    {
        public string method, url, protocol;
        public Dictionary<string, string> headers;
        public Request(StreamReader sr)
        {
            string line = sr.ReadLine();
            string[] p = line.Split(' ');
            method = p[0];
            url = p[1];
            protocol = p[2];

            headers = new Dictionary<string, string>();
            while (!String.IsNullOrEmpty(line = sr.ReadLine()))
            {
              //  Console.WriteLine(line);

                int i = line.IndexOf(":");
                if (i >= 0)
                {
                    headers.Add(line.Substring(0, i), line.Substring(i + 1));
                }
                else
                {
                    headers.Add(line, "");
                }
            }

        }
    }
    public class Response
    {
        public string status = "200 OK";
        public class ContentType
        {
            public static string text = "text/plain";
            public static string png = "image/png";
            public static string json = "application/json";
        }
        public string contentType = ContentType.json;
        public Dictionary<string, string> headers = new Dictionary<string, string>();
        public byte[] data = new byte[] { };
    }
    class Program
    {
        private Thread serverThread;
        TcpListener listener;

        public void start(int port )
        {
            IPAddress ipAddr = IPAddress.Any;
            listener = new TcpListener(ipAddr, port);
            serverThread = new Thread(() =>
            {
                listener.Start();
                while (true)
                {
                    Socket s = listener.AcceptSocket();
                    ThreadPool.QueueUserWorkItem(threadProc, s);
                }
            });
            serverThread.Start();
        }

        private Response reqProc(Request req)
        {
            if (req.url == "/")
            {
                return getAllFonts();
            }
            else
            {
                return getImage(req.url);
            }
        }

        private void threadProc(object obj)
        {
            Socket s = (Socket)obj;
            NetworkStream ns = new NetworkStream(s);
            s.ReceiveTimeout = 30;
            s.SendTimeout = 30;
            ns.ReadTimeout = 30;
            ns.WriteTimeout = 30;
            try
            {
                StreamReader sr = new StreamReader(ns);

                Console.WriteLine("{0} {1}", DateTime.Now.ToString(), (s.RemoteEndPoint as IPEndPoint).Address);
                Request req = new Request(sr);
                //Console.WriteLine("req");

                Response resp = reqProc(req);
                //Console.WriteLine("resp");

                StreamWriter sw = new StreamWriter(ns);
                sw.WriteLine("HTTP/1.0 {0}", resp.status);
                sw.WriteLine("Content-Type: {0}", resp.contentType);
                foreach (string k in resp.headers.Keys)
                    sw.WriteLine("{0}: {1}", k, resp.headers[k]);
                sw.WriteLine("Content-Length: {0}", resp.data.Length);
                sw.WriteLine();
                sw.Flush();
                s.Send(resp.data);
                s.Shutdown(SocketShutdown.Both);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);

            }
            finally
            {
                ns.Close();
            }

        }


        public void stop()
        {
            listener.Stop();
            serverThread.Abort();
        }

        Response getAllFonts()
        {
            Dictionary<String, String[]> fonts = new Dictionary<string, string[]>();
            fonts.Add("eng", Utils.engFontNames);
            fonts.Add("chi", Utils.chiFontNames);
            Response resp = new Response();
            resp.contentType = Response.ContentType.json;
            DataContractJsonSerializer json = new DataContractJsonSerializer(fonts.GetType(), 
                new DataContractJsonSerializerSettings() { UseSimpleDictionaryFormat = true});
            
            MemoryStream ms = new MemoryStream();
            json.WriteObject(ms, fonts);
            resp.data = ms.ToArray();
            return resp;
        }

        Response getImage(string url)
        {
            var nvc = HttpUtility.ParseQueryString(url.Substring(url.IndexOf("?")+1));
            IDictionary<String, String> param = nvc.AllKeys.ToDictionary(k=>k,k=>nvc[k]);
            String text = param["text"];
            String fontname = param["fontname"];
            int fontsize = int.Parse(param["fontsize"]);
            int fontmode = int.Parse(param["fontmode"]);
            int fonthint = int.Parse(param["fonthint"]);
            Bitmap image = Utils.Render(text, fontname, fontsize, fontmode, fonthint);
            Response resp = new Response();
            resp.contentType = Response.ContentType.png;
            MemoryStream ms = new MemoryStream();
            image.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
            resp.data = ms.ToArray();
            return resp;
        }
        static void Main(string[] args)
        {
            Program p = new Program();
            p.start(8888);
            Console.WriteLine("press any key to exit.");
            Console.Read();
            p.stop();
        }
    }
}
