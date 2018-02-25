

#ifndef __MYNET_HPP__
#define __MYNET_HPP__

#include "myBlob.hpp"
#include "myLayer.hpp"
//#include "test.hpp"
#include <json/json.h>
#include <unordered_map>
#include <fstream>
#include "gordon.NetParameter.pb.h"
#include <iostream>
#include <vector>
using std::unordered_map;
using std::shared_ptr;

struct NetParam
{
	/*! methods of update net parameters, sgd/momentum/... */
	std::string update;  //优化算法
	/*! learning rate */
	double lr;
	double lr_decay;
	/*! momentum parameter */
	double momentum;
	int num_epochs;
	/*! whether use batch size */
	bool use_batch;
	int batch_size;
	/*! regulazation parameter */
	double reg;
	/*! \brief acc_frequence, how many iterations to check val_acc and train_acc */
	int acc_frequence;
	bool acc_update_lr;
	/* 是否保存模型快照；快照保存间隔*/
	bool snap_shot;
	int snapshot_interval;
	/* 是否采用fine-tune方式训练；预训练模型文件.gordonmodel所在路径*/
	bool fine_tune;
	std::string preTrainModel;

	vector<std::string> layers;
	vector<std::string> ltypes;
	unordered_map<std::string, Param> params;
	void readNetParam(std::string file);
};

class Net
{

public:
	Net(){}
	void trainNet(shared_ptr<Blob>& X,shared_ptr<Blob>& Y,NetParam& param,std::string mode = "fb");
	//void testNet(NetParam& param);
	void initNet(NetParam& param, vector<shared_ptr<Blob>>& X, vector<shared_ptr<Blob>>& Y);
	void train(NetParam& param);
	void PromptForParam(gordon::LayerDictionary* dict);
	void ListParam(const gordon::NetParameter& net_param);
	/////////void sampleInitData();
	//void testLayer(NetParam& param, int lnum);

private:
	//void _test_fc_layer(vector<shared_ptr<Blob>>& in,vector<shared_ptr<Blob>>& grads,shared_ptr<Blob>& dout);
	//void _test_conv_layer(vector<shared_ptr<Blob>>& in,vector<shared_ptr<Blob>>& grads,shared_ptr<Blob>& dout,Param& param);
	//void _test_pool_layer(vector<shared_ptr<Blob>>& in,vector<shared_ptr<Blob>>& grads,shared_ptr<Blob>& dout,Param& param);
	//void _test_relu_layer(vector<shared_ptr<Blob>>& in,vector<shared_ptr<Blob>>& grads,shared_ptr<Blob>& dout);
	//void _test_dropout_layer(vector<shared_ptr<Blob>>& in,vector<shared_ptr<Blob>>& grads,shared_ptr<Blob>& dout,Param& param);
	//void _test_svm_layer(vector<shared_ptr<Blob>>& in,shared_ptr<Blob>& dout);
	//void _test_softmax_layer(vector<shared_ptr<Blob>>& in,shared_ptr<Blob>& dout);

	vector<std::string> layers_;
	vector<std::string> ltype_;
	double loss_;
	// 训练集
	shared_ptr<Blob> X_train_;
	shared_ptr<Blob> Y_train_;
	// 验证集
	shared_ptr<Blob> X_val_;
	shared_ptr<Blob> Y_val_;

	unordered_map<std::string, vector<shared_ptr<Blob>>> data_;  //<layer名称 , Blob集合（输入数据A_prev, 卷积核W, 偏置b）>
	unordered_map<std::string, vector<shared_ptr<Blob>>> grads_;//<layer类型 , Blob集合（ dA，dW，db）>
	unordered_map<std::string, vector<shared_ptr<Blob>>> num_grads_;

	std::string type_;

	vector<double> loss_history_;
	vector<double> train_acc_history_;
	vector<double> val_acc_history_;

	unordered_map<std::string, vector<shared_ptr<Blob>>> step_cache_;
	unordered_map<std::string, vector<shared_ptr<Blob>>> best_model_;   //模型快照！

}; // class Net

#endif //__MYNET_HPP__
