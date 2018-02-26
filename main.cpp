#include "gordon.NetParameter.pb.h"
#include "gordon_cnn.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <json/json.h>  //还得安装json文件解析库才行
using namespace std;
using namespace arma;
//作者：gordon
/*
魔数的概念：很多类型的文件，其起始的几个字节的内容是固定的（或是有意填充，或是本就如此）。
根据这几个字节的内容就可以确定文件类型，因此这几个字节的内容被称为魔数 (magic number)。
大端存储与小端存储：一个博客里面写的十分详细，总体来说：大端存储类似人的正常思维，小端存储机器处理更方便
小端：较高的有效字节存放在较高的的存储器地址，较低的有效字节存放在较低的存储器地址。
大端：较高的有效字节存放在较低的存储器地址，较低的有效字节存放在较高的存储器地址。
mnist原始数据文件中32位的整型值是大端存储，C/C++变量是小端存储，所以读取数据的时候，需要对其进行大小端转换。
关于用c++解析mnist可参考博客：
http://blog.csdn.net/lhanchao/article/details/53503497
http://blog.csdn.net/sheng_ai/article/details/23267039
具体的格式及介绍见官网 ：
http://yann.lecun.com/exdb/mnist/
注意：只有文件头的个别数字需要大小端转换，其余的60000个有效数据则不需要。
*/

int ReverseInt(int i)  ////把大端数据转换为我们常用的小端数据 （大小端模式的原因，可能Lecun制作数据集用的大小端跟我们的机器不一样）
{
	unsigned char ch1, ch2, ch3, ch4;  //一个int有4个char
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMnistLabel(string path, shared_ptr<Blob>& label) {
	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		//1.从文件中获知魔术数字，图片数量
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		cout << "magic_number=" << magic_number << endl;
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		cout << "number_of_Labels=" << number_of_images << endl;
		//2.将所有标签转为Blob存储！（手写数字识别：0~9）
		for (int i = 0; i<number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			//cout << "Label=" << (int)temp << endl;
			//存入Blob！【Blob实际上是一个vector<cube>向量，cube(height索引, width索引, chanel索引)】
			//下面 (*label)[i](0, 0, (int)temp) 表示第i个cube的(0,0)点处的temp通道【注意：在main初始化存储label的cube是1行1列10通道（深度）】
			(*label)[i](0, 0, (int)temp) = 1;  //“one hot”形式的标签表示。即索引0~9哪个置1则标签为几！//注意：cube格式为(h, w, c)
		}
	}
	else {
		cout << "no label file found :-(" << endl;
	}
}

void ReadMnistData(string path, shared_ptr<Blob>& image)
{
	ifstream file(path, ios::binary);  //数据路径列表文件（json文件）
	if (file.is_open())
	{
		//mnist原始数据文件中32位的整型值是大端存储，C/C++变量是小端存储，所以读取数据的时候，需要对其进行大小端转换!!!!
		//1.从文件中获知魔术数字（一般都是起到标识的作用，没有其他实质性的用处），图片数量和图片宽高信息
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);  //高低字节调换
		cout << "magic_number=" << magic_number << endl;
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		cout << "number_of_images=" << number_of_images << endl;
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		cout << "n_rows=" << n_rows << endl;
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		cout << "n_cols=" << n_cols << endl;

		//2.将图片转为Blob存储！
		for (int i = 0; i<number_of_images; ++i)  //遍历所有图片
		{
			for (int r = 0; r<n_rows; ++r)   //遍历高
			{
				for (int c = 0; c<n_cols; ++c)   //遍历宽
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));      //读入一个像素值！		
					//可视化数据
					//double tmp = (double)temp / 255;
					//if (tmp != 0)tmp = 1;
					//cout << tmp << " ";
					//if (c == 27)cout << endl;		 //第一个数字是5（后面0、4、1、9）
					//归一化后存入Blob！【Blob实际上是一个vector<cube>向量，cube(height索引, width索引, chanel索引)】
					(*image)[i](r, c, 0) = (double)temp / 255;    //注意：cube格式为(h, w, c)
				}
			}
		}
	}
	else
	{
		cout << "no data file found :-(" << endl;
	}
}

void trainMnist(shared_ptr<Blob>& X, shared_ptr<Blob>& Y, string config)
{
	NetParam param;
	param.readNetParam(config);  //读取网络结构配置参数（config为json配置文件所在路径）

	shared_ptr<Blob> X_train(new Blob(X->subBlob(0, 59000))); //训练集数据
	shared_ptr<Blob> Y_train(new Blob(Y->subBlob(0, 59000))); //训练集标签
	shared_ptr<Blob> X_val(new Blob(X->subBlob(59000, 60000)));//验证集数据
	shared_ptr<Blob> Y_val(new Blob(Y->subBlob(59000, 60000)));//验证集标签
	vector<shared_ptr<Blob>> XX{ X_train, X_val }; //数据集
	vector<shared_ptr<Blob>> YY{ Y_train, Y_val }; //标签集
	vector<std::string> ltypes_ = param.ltypes;
	vector<std::string> layers_ = param.layers;
	for (int i = 0; i < ltypes_.size(); ++i)
	{
		cout << "ltype = " << ltypes_[i] << endl;
	}
	for (int i = 0; i < layers_.size(); ++i)
	{
		cout << "layer = " << layers_[i] << endl;
	}
	Net inst;
	//初始化网络结构！
	inst.initNet(param, XX, YY); 
	////////inst.testNet(param);  //对整个网络（逐层）做梯度检验
	//开始训练！
	inst.train(param);   //（测试的话也可以用这个函数，利用里面trainNet函数的mode=forward参数控制就可以，或者自己专门写一个测试的函数，不难）
	
}


int main(int argc, char** argv)
{

	//shared_ptr<Blob> images(new Blob(10000, 1, 28, 28));  //创建一个Blob，cube数量10000，每个cube为通道1，28行28列
	//ReadMnistData("mnist_data/test/t10k-images.idx3-ubyte", images);  //读取data
	//vector<cube> lists = images->get_data();
	//for (int i = 0; i<3; ++i)  //遍历所有图片
	//{
	//	for (int r = 0; r<28; ++r)   //遍历高
	//	{
	//		for (int c = 0; c<28; ++c)   //遍历宽
	//		{
	//			double tmp = lists[i](r, c, 0);  //打印每一个像素
	//			if (tmp != 0)tmp = 1;
	//			cout << tmp << " ";
	//			if (c == 27)cout << endl;		 //第一个数字是5（后面0、4、1、9）
	//		}
	//	}
	//}
	//-------------------------------------------------------------------------------------

	shared_ptr<Blob> images(new Blob(60000, 1, 28, 28));  //创建一个Blob，cube数量10000，每个cube为通道1，28行28列
	shared_ptr<Blob> labels(new Blob(60000, 10, 1, 1, TZEROS));//创建一个Blob，cube数量10000，每个cube为通道（深度）10，1行1列  [10000,10,1,1]
	ReadMnistData("mnist_data/train/train-images.idx3-ubyte", images);  //读取data
	ReadMnistLabel("mnist_data/train/train-labels.idx1-ubyte", labels);   //读取label
	trainMnist(images, labels, "mnist.json");  //开始训练

	 return 0;
}




