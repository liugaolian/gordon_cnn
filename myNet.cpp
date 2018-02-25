#include "myNet.hpp"
using namespace std;

void NetParam::readNetParam(std::string file) 
{
	std::ifstream ifs;
	ifs.open(file);
	assert(ifs.is_open());
	Json::Reader reader;
	Json::Value value;
	if (reader.parse(ifs, value))
	{
		if (!value["train"].isNull()) 
		{
			auto &tparam = value["train"];
			this->lr = tparam["learning rate"].asDouble(); //解析成Double类型存放
			this->lr_decay = tparam["lr decay"].asDouble();
			this->update = tparam["update method"].asString();//解析成String类型存放
			this->momentum = tparam["momentum parameter"].asDouble();
			this->num_epochs = tparam["num epochs"].asInt();//解析成Int类型存放
			this->use_batch = tparam["use batch"].asBool();//解析成Bool类型存放
			this->batch_size = tparam["batch size"].asInt();
			this->reg = tparam["reg"].asDouble();
			this->acc_frequence = tparam["acc frequence"].asInt();
			this->acc_update_lr = tparam["frequence update"].asBool();
			this->snap_shot = tparam["snapshot"].asBool();
			this->snapshot_interval = tparam["snapshot interval"].asInt();
			this->fine_tune = tparam["fine tune"].asBool();
			this->preTrainModel = tparam["pre train model"].asString();//解析成String类型存放
		}
		if (!value["net"].isNull()) 
		{
			auto &nparam = value["net"];
			for (int i = 0; i < (int)nparam.size(); ++i) 
			{
				auto &ii = nparam[i];
				this->layers.push_back(ii["name"].asString());
				this->ltypes.push_back(ii["type"].asString());
				if (ii["type"].asString() == "Conv")
				{
					int num = ii["kernel num"].asInt();
					int width = ii["kernel width"].asInt();
					int height = ii["kernel height"].asInt();
					int pad = ii["pad"].asInt();
					int stride = ii["stride"].asInt();
					this->params[ii["name"].asString()].setConvParam(stride, pad, width, height, num);
				}
				if (ii["type"].asString() == "Pool") 
				{
					int stride = ii["stride"].asInt();
					int width = ii["kernel width"].asInt();
					int height = ii["kernel height"].asInt();
					this->params[ii["name"].asString()].setPoolParam(stride, width, height);
				}
				if (ii["type"].asString() == "Fc") 
				{
					int num = ii["kernel num"].asInt();
					this->params[ii["name"].asString()].fc_kernels = num;
				}
			}
		}
	}
}



	void Net::trainNet(shared_ptr<Blob>& X,   //X_batch（mini-batch）
		shared_ptr<Blob>& Y,   //Y_batch（mini-batch）
		NetParam& param,
		std::string mode)  //mode用于控制用于训练还是用于测试！
	{
		/*! fill X, Y */
		data_[layers_[0]][0] = X;         //将本轮mini-batch填充进第零层的输入Blob中！（A_prev）
		data_[layers_.back()][1] = Y;   //将本轮mini-batch（labels）填充进最末层的标签Blob中！（Y），data_[layers_.back()][0] 为输入Blob！

		// debug
		Blob pb, pd;

		/* step1. forward 开始前向传播！！！*/
		int n = ltype_.size();  //层类型个数（=层名个数）
		for (int i = 0; i < n - 1; ++i)   //遍历层
		{
			std::string ltype = ltype_[i];       //当前层类型
			std::string lname = layers_[i];   //当前层名（跟层类型的索引是逐一对应的）
			shared_ptr<Blob> out;             //定义一个输出Blob，保存该层前向计算所得的输出
			if (ltype == "Conv")
			{
				int tF = param.params[lname].conv_kernels;   //卷积核个数（由层名称索引得到）
				int tC = data_[lname][0]->get_C();                    //卷积核深度（=输入Blob深度）
				int tH = param.params[lname].conv_height;   //卷积核高
				int tW = param.params[lname].conv_width;    //卷积核宽  
				/*--------------------  初始化卷积核W和偏置b（也就第一次forward时会执行）  ----------------------*/
				if (!data_[lname][1])   //若本conv层的卷积核Blob为空，则创建（若是有预训练权重加载，则不会执行初始化）
				{
					cout << "------ Init weights  with Gaussian ------" << endl;
					data_[lname][1].reset(new Blob(tF, tC, tH, tW, TRANDN)); //将空Blob重设为（tF, tC, tH, tW）
					(*data_[lname][1]) *= 1e-2;  //初始化：卷积核乘以0.01（所有权值逐一乘以0.01）
				}
				if (!data_[lname][2])    //若本conv层的偏置Blob为空，则创建（若是有预训练bias加载，则不会执行初始化）
				{
					cout << "------ Init bias with Gaussian ------" << endl;
					data_[lname][2].reset(new Blob(tF, 1, 1, 1, TRANDN)); //将空Blob重设为（tF, 1, 1, 1）
					(*data_[lname][2]) *= 1e-1;  // 初始化：bias乘以0.01（所有bias逐一乘以0.1）
				}
				/*------  该卷积层开始做前向传播！  -------*/
				ConvLayer::forward(data_[lname], out, param.params[lname]);
			}
			if (ltype == "Pool")
			{
				PoolLayer::forward(data_[lname], out, param.params[lname]);
				pb = *data_[lname][0];
			}
			if (ltype == "Fc")
			{
				int tF = param.params[lname].fc_kernels;
				int tC = data_[lname][0]->get_C();
				int tH = data_[lname][0]->get_H();
				int tW = data_[lname][0]->get_W();
				if (!data_[lname][1])
				{
					data_[lname][1].reset(new Blob(tF, tC, tH, tW, TRANDN));
					(*data_[lname][1]) *= 1e-2;
				}
				if (!data_[lname][2])
				{
					data_[lname][2].reset(new Blob(tF, 1, 1, 1, TRANDN));
					(*data_[lname][2]) *= 1e-1;
				}
				AffineLayer::forward(data_[lname], out);
			}
			if (ltype == "Relu")
			{
				ReluLayer::forward(data_[lname], out);
			}
			if (ltype == "Dropout")
			{
				DropoutLayer::forward(data_[lname], out, param.params[lname]);
			}
			//------------------------------------------------------
			data_[layers_[i + 1]][0] = out;   //将该层输出Blob填充进下一层的输入Blob，继续执行前向计算，直至Softmax层的前一层！
			//------------------------------------------------------
		}

		/* step2. 最末层softmax做前向传播和计算代价！*/
		std::string loss_type = ltype_.back(); //最末层的层类型
		shared_ptr<Blob> dout;  //定义误差信号！（即损失梯度，用来方向传播的）
		if (loss_type == "SVM")
			SVMLossLayer::go(data_[layers_.back()], loss_, dout);
		if (loss_type == "Softmax")
			SoftmaxLossLayer::go(data_[layers_.back()], loss_, dout);  //最末层：Softmax层的前向传播，可以得到本次forward损失值loss_和损失梯度dout
		grads_[layers_.back()][0] = dout;

		loss_history_.push_back(loss_); //叠加进loss向量中（可以打印输出，也可以把每个迭代周期的损失描点作图）

		if (mode == "forward")  //如果仅用于前向传播（做测试，不训练），则提前退出！
			return;

		/* step3. backward 开始反向传播！*/
		for (int i = n - 2; i >= 0; --i)  //从最末层之前的一层（即除去softmax），即全连接层开始逐层反向传播
		{
			std::string ltype = ltype_[i];
			std::string lname = layers_[i];
			if (ltype == "Conv")
			{
				//输入参数：Relu层得到的损失梯度（误差信号） -- conv层A_prev，W, b  --  conv层反向计算得到的dA，dW, db -- conv层kernel宽高和步长参数    
				ConvLayer::backward(grads_[layers_[i + 1]][0], data_[lname], grads_[lname], param.params[lname]);
			}
			if (ltype == "Pool")
			{
				//输入参数：fc层得到的损失梯度（误差信号） -- pool层A_prev，W, b  --  pool层反向计算得到的dA，dW, db -- pool层宽高和步长参数    
				PoolLayer::backward(grads_[layers_[i + 1]][0], data_[lname], grads_[lname], param.params[lname]);
			}
			if (ltype == "Fc")
			{
				//输入参数：softmax得到的dout （误差信号）-- 全连接层A_prev，W, b -- 全连接层反向计算得到的dA，dW, db
				AffineLayer::backward(grads_[layers_[i + 1]][0], data_[lname], grads_[lname]);
			}
			if (ltype == "Relu")
			{
				//输入参数：pool层得到的损失梯度 （误差信号）-- Relu层A_prev，W, b -- Relu层反向计算得到的dA，dW, db
				ReluLayer::backward(grads_[layers_[i + 1]][0], data_[lname], grads_[lname]);
			}
		}

		// regularition 正则化！
		double reg_loss = 0;
		for (auto i : layers_)  //遍历层（c++11语法）
		{
			if (grads_[i][1])  //该层权值梯度不为空
			{
				// it's ok?                       //-----------gordon：嗯？作者做这个注释啥意思？难道他也不确定这段代码行不行？！
				Blob reg_data = param.reg * (*data_[i][1]);
				(*grads_[i][1]) = (*grads_[i][1]) + reg_data;
				reg_loss += data_[i][1]->sum();
			}
		}
		reg_loss *= param.reg * 0.5;
		loss_ = loss_ + reg_loss;

		return;
	}

	//void Net::testNet(NetParam& param) 
	//{
	//	shared_ptr<Blob> X_batch(new Blob(X_train_->subBlob(0, 1)));
	//	shared_ptr<Blob> Y_batch(new Blob(Y_train_->subBlob(0, 1)));
	//	trainNet(X_batch, Y_batch, param);
	//	cout << "BEGIN TEST LAYERS" << endl;
	//	for (int i = 0; i < (int)layers_.size(); ++i)  //遍历层，逐层做梯度检验
	//	{
	//		testLayer(param, i);  //单层梯度检验，参数：网络参数；第几层
	//		printf("\n");
	//	}
	//}

	void Net::ListParam(const gordon::NetParameter& net_param)
	{
		cout << "net_param.dict_size() = " << net_param.dict_size() << endl;
		for (int i = 0; i < net_param.dict_size(); i++)  //遍历所有字典（每生成一个模型参数快照，就会多一个字典）
		{
			const gordon::LayerDictionary& dict = net_param.dict(i);
			for (int j = 0; j < dict.blobs_size(); j++)  //遍历整个字典中的所有参数Blob
			{
				const gordon::LayerDictionary::ParamBlob& blob = dict.blobs(j);
				cout << "LayerName = " << blob.lname() << "     LayerType = " << blob.type() << endl;
				cout << "Blob(N,C,H,W) = （" << blob.cube_num() << "," << blob.cube_ch() << "," << blob.cube_size() << "," << blob.cube_size() << "）" << endl;
				int number = 0;
				shared_ptr<Blob> param_blob(new Blob(blob.cube_num(), blob.cube_ch(), blob.cube_size(), blob.cube_size()));
				//cout << "-----------------" << endl;
				for (int i = 0; i<blob.cube_num(); ++i)  //遍历该Blob所有卷积核
				{
					for (int c = 0; c < blob.cube_ch(); ++c)  //遍历深度
					{
						for (int h = 0; h<blob.cube_size(); ++h)   //遍历高
						{
							for (int w = 0; w<blob.cube_size(); ++w)   //遍历宽
							{
								const gordon::LayerDictionary::ParamBlob::ParamValue& param_val = blob.param_val(number);
								//cout << param_val.val() << " ";
								(*param_blob)[i](h, w, c) = param_val.val();
								number++;
							}
							//cout << endl;
						}
						//cout << "---------------" << endl;
					}
					//cout << "= = = = = = = = = = = = = = = = = = = = = = = " << endl;
				}
				//cout << "param number = " << number << endl;
				//这里要实现：将(*param_blob)内容保存至叠加进data_，将次函数移动到initNet()中会不会比较好？
				string layer_name = blob.lname();
				if (blob.type()==0)
					data_[layer_name][1] = param_blob;  //权值
				else
					data_[layer_name][2] = param_blob;  //偏置
			}
			cout << "-----------------" << endl;
		}
		cout << endl << "/////////////////////////////// 检验data_ /////////////////////////////////" << endl << endl;
		for (auto lname : layers_)  //遍历层（c++11语法）
		{
			for (int j = 1; j <= 2; ++j)   //j为1时更新该层权值W,  j为2时更新该层偏置b
			{
				if (!data_[lname][1] || !data_[lname][2])   //若该层输入Blob的权重和bias有一个为空（此时可能为relu层和pool，没有参数的，所以要跳过）
				{
					continue;  //跳过本轮循环，重新执行循环（注意不是break跳出循环）
				}
				int number = 0;
				int blob_num = data_[lname][j]->get_N();  //该Blob中cube（卷积核）数量
				int blob_ch = data_[lname][j]->get_C();     //cube的深度
				int blob_height = data_[lname][j]->get_H();  //cube的高
				int blob_width = data_[lname][j]->get_W();  //cube的宽
				if (j == 1)
					cout << "LayerName = " << lname << "     LayerType = 0 (weight)" << endl;
				else
					cout << "LayerName = " << lname << "     LayerType = 1 (bias)" << endl;
				cout << "Blob(N,C,H,W) = （" << blob_num << "," << blob_ch << "," << blob_height << "," << blob_width << "）" << endl;
				vector<cube> lists = data_[lname][j]->get_data();  //获得weight
				assert(lists.size() == blob_num);
				cout << "-------------------------------------" << endl;
				for (int i = 0; i<blob_num; ++i)  //遍历所有卷积核
				{
					for (int c = 0; c < blob_ch; ++c)  //遍历深度
					{
						for (int h = 0; h<blob_height; ++h)   //遍历高
						{
							for (int w = 0; w<blob_width; ++w)   //遍历宽
							{
								double tmp = lists[i](h, w, c);  //打印每一个参数值
								cout << tmp << "  ";
								number++;
							}
							cout << endl;
						}
						cout << "-------------------------------------" << endl;
					}
					cout << "= = = = = = = = = = = = = =  = = = = = = = = = " << endl;
				}
				cout << "param number = " << number << endl;
			}
		}
	}

	void Net::initNet(NetParam& param, vector<shared_ptr<Blob>>& X, vector<shared_ptr<Blob>>& Y)
	{
		layers_ = param.layers;   // 层名，param.layers类型为vector<string>
		ltype_ = param.ltypes;    // 层类型 , param.ltypes类型为vector<string>
		for (int i = 0; i < (int)layers_.size(); ++i)   //遍历每一层
		{
			data_[layers_[i]] = vector<shared_ptr<Blob>>(3);    //初始化该层的前向传播所需变量（大小为3个vector）:      A_prev, W, b
			grads_[layers_[i]] = vector<shared_ptr<Blob>>(3);  //初始化该层的反向梯度（大小为3个vector）:      dA，dW，db
			step_cache_[layers_[i]] = vector<shared_ptr<Blob>>(3);  //初始化该层用于backward的缓存
			best_model_[layers_[i]] = vector<shared_ptr<Blob>>(3); //初始化最终的模型输出变量？！保存模型快照？？！！
		}
		X_train_ = X[0];  //初始化训练集数据
		Y_train_ = Y[0]; //初始化训练集标签
		X_val_ = X[1];   //初始化验证集数据
		Y_val_ = Y[1];   //初始化验证集标签

		if (param.fine_tune)//是重新训练还是fine-tune取决于data_[lname][1]和data_[lname][2]是否有被填充过值
		{
			GOOGLE_PROTOBUF_VERIFY_VERSION;
			gordon::NetParameter net_param;
			{
				//fstream input("./iter20.gordonmodel", ios::in | ios::binary);
				fstream input(param.preTrainModel, ios::in | ios::binary);
				if (!input)
				{
					cout << param.preTrainModel << " was not found ！！！" << endl;
					return;
				}
				else
				{
					if (!net_param.ParseFromIstream(&input))
					{
						cerr << "Failed to parse the " << param.preTrainModel << " ！！！" << endl;
						return;
					}
					cout <<"--- Load the"<< param.preTrainModel << " sucessfully ！！！---" << endl;
				}
			}
			ListParam(net_param);
			google::protobuf::ShutdownProtobufLibrary();
			cout << "--------------- You will fine-tune a model --------------" << endl;
		}
		cout << "--------------- InitNet Done --------------" << endl;
		return;
	}

	void Net::PromptForParam(gordon::LayerDictionary* dict)
	{
		for (auto lname : layers_)  //遍历层（c++11语法）
		{
			for (int j = 1; j <= 2; ++j)   //j为1时更新该层权值W,  j为2时更新该层偏置b
			{
				if (!data_[lname][1] || !data_[lname][2])   //若该层输入Blob的权重和bias有一个为空（此时可能为relu层和pool，没有参数的，所以要跳过）
				{
					continue;  //跳过本轮循环，重新执行循环（注意不是break跳出循环）
				}
				int number = 0;
				int blob_num = data_[lname][j]->get_N();  //该Blob中cube（卷积核）数量
				int blob_ch = data_[lname][j]->get_C();     //cube的深度
				int blob_height = data_[lname][j]->get_H();  //cube的高
				int blob_width = data_[lname][j]->get_W();  //cube的宽
				if (j == 1)
					cout << "LayerName = " << lname << "     LayerType = 0 (weight)" << endl;
				else 
					cout << "LayerName = " << lname << "     LayerType = 1 (bias)" << endl;
				cout << "Blob(N,C,H,W) = （" << blob_num << "," << blob_ch << "," << blob_height << "," << blob_width << "）" << endl;
				vector<cube> lists = data_[lname][j]->get_data();  //获得weight
				assert(lists.size() == blob_num);
				//创建一个Blob存储weight
				gordon::LayerDictionary::ParamBlob* param_blob = dict->add_blobs();
				param_blob->set_cube_num(blob_num);
				param_blob->set_cube_size(blob_height);
				param_blob->set_cube_ch(blob_ch);
				//写入该层层名
				if (!lname.empty())
					param_blob->set_lname(lname);
				if (j == 1)
					param_blob->set_type(gordon::LayerDictionary::WEIGHT);
				else
					param_blob->set_type(gordon::LayerDictionary::BIAS);

				cout << "-------------------------------------" << endl;
				for (int i = 0; i<blob_num; ++i)  //遍历所有卷积核
				{
					for (int c = 0; c < blob_ch; ++c)  //遍历深度
					{
						for (int h = 0; h<blob_height; ++h)   //遍历高
						{
							for (int w = 0; w<blob_width; ++w)   //遍历宽
							{
								double tmp = lists[i](h, w, c);  //打印每一个参数值
								cout << tmp << "  ";
								gordon::LayerDictionary::ParamBlob::ParamValue* param_val = param_blob->add_param_val();
								param_val->set_val(tmp);
								number++;
							}
							cout << endl;
						}
						cout << "-------------------------------------" << endl;
					}
					cout << "= = = = = = = = = = = = = =  = = = = = = = = = " << endl;
				}
				cout << "param number = " << number << endl;
			}
		}
	}

	void Net::train(NetParam& param)
	{
		// to be delete
		int N = X_train_->get_N();  //获取训练集数据总数
		cout << "N = " << N << endl;
		int iter_per_epochs;     //单个epoch的批次数    //59000/200 = 295
		if (param.use_batch)
		{
			iter_per_epochs = N / param.batch_size;  // 单个epoch的批次数 = 数据总数 / 每批次数据个数 （mini-batch的个数） 59000/200=295
		}
		else
		{
			iter_per_epochs = N;  //不分批次训练，直接全部送进去前向运算后再执行一次BP
		}
		int num_iters = iter_per_epochs * param.num_epochs;   //总的批次数 = 单个epoch的批次数 * epoch总数（一个epoch走完一遍训练集）5x295=1475
		int epoch = 0;
		cout << "num_iters = " << num_iters << endl;
		//创建一个网络参数对象
		GOOGLE_PROTOBUF_VERIFY_VERSION;
		gordon::NetParameter net_param;

		// iteration 开始迭代！
		//for (int iter = 0; iter < num_iters; ++iter)    //遍历每一个批次（mini-batch）
		for (int iter = 0; iter < 25; ++iter)    //debug:gordon
		{
			// batch
			shared_ptr<Blob> X_batch;
			shared_ptr<Blob> Y_batch;
			if (param.use_batch)   //采用mini-batch做梯度下降，需要把训练集分成若干mini-batch
			{
				// 注意：深拷贝！
				//a. 在整个训练集Blob中截取单批次训练数据
				X_batch.reset(new Blob(X_train_->subBlob((iter * param.batch_size) % N,  /*已完成批次 x 单批次数据量 = 已完成数据量；已完成数据量 % 数据总量 = 已完成比例（待截取起点位置）*/
					((iter + 1) * param.batch_size) % N)));   /*下一批次（待截取终点位置）*/
				//b. 在整个验证集Blob中截取单批次验证数据
				Y_batch.reset(new Blob(Y_train_->subBlob((iter * param.batch_size) % N,
					((iter + 1) * param.batch_size) % N)));
			}
			else   //不采用mini-batch做梯度下降，直接全部数据前向计算完后才做一次权重更新
			{
				shared_ptr<Blob> X_batch = X_train_;  //不截取，直接全部数据（不分批次）
				shared_ptr<Blob> Y_batch = Y_train_;
			}

			// step1. 用刚刚截取的该批次数据（或全部数据）训练网络  ---->前向计算和反向计算
			trainNet(X_batch, Y_batch, param);  //是重新训练还是fine-tune取决于data_[lname][1]和data_[lname][2]是否有被填充过值

			// step2. update  参数更新！！！
			for (int i = 0; i < (int)layers_.size(); ++i)  //遍历层
			{
				std::string lname = layers_[i];
				if (!data_[lname][1] || !data_[lname][2]) //若该层输入Blob的权重和bias有一个为空（此时可能为relu层和pool，没有参数的，所以要跳过）
				{
					continue;  //跳过本轮循环，重新执行循环（注意不是break跳出循环）
				}
				for (int j = 1; j <= 2; ++j)   //j为1时更新该层权值W,  j为2时更新该层偏置b
				{
					assert(param.update == "momentum" ||
						param.update == "rmsprop" ||
						param.update == "adagrad" ||
						param.update == "sgd");   //断言：优化算法属于这四种的一种，可以查出字符串输入造成的错误！
					shared_ptr<Blob> dx(new Blob(data_[lname][j]->size()));

					//多种优化算法可供选择，具体原理可看Ng的课程，讲得比较详细了！！！
					if (param.update == "sgd")  //随机梯度下降
					{
						*dx = -param.lr * (*grads_[lname][j]);//j为1时更新该层权值W,  j为2时更新该层偏置b
					}
					if (param.update == "momentum")   //添加动量的梯度下降
					{
						if (!step_cache_[lname][j])
						{
							step_cache_[lname][j].reset(new Blob(data_[lname][j]->size(), TZEROS));
						}
						Blob ll = param.momentum * (*step_cache_[lname][j]);
						Blob rr = param.lr * (*grads_[lname][j]);
						*dx = ll - rr;
						step_cache_[lname][j] = dx;
					}
					if (param.update == "rmsprop")    //rmsprop梯度下降
					{
						// change it self
						double decay_rate = 0.99;
						if (!step_cache_[lname][j])
						{
							step_cache_[lname][j].reset(new Blob(data_[lname][j]->size(), TZEROS));
						}
						Blob r1 = decay_rate * (*step_cache_[lname][j]);
						Blob r2 = (1 - decay_rate) * (*grads_[lname][j]);
						Blob r3 = r2 * (*grads_[lname][j]);
						*step_cache_[lname][j] = r1 + r3;
						Blob d1 = (*step_cache_[lname][j]) + 1e-8;
						Blob u1 = param.lr * (*grads_[lname][j]);
						Blob d2 = sqrt(d1);
						Blob r4 = u1 / d2;
						*dx = 0 - r4;
					}
					if (param.update == "adagrad")  //adagrad，它的优化版本是rmsprop（注意不是Adam，试一下自己实现？）
					{
						if (!step_cache_[lname][j])
						{
							step_cache_[lname][j].reset(new Blob(data_[lname][j]->size(), TZEROS));
						}
						*step_cache_[lname][j] = (*grads_[lname][j]) * (*grads_[lname][j]);
						Blob d1 = (*step_cache_[lname][j]) + 1e-8;
						Blob u1 = param.lr * (*grads_[lname][j]);
						Blob d2 = sqrt(d1);
						Blob r4 = u1 / d2;
						*dx = 0 - r4;
					}
					//提示：j为1时更新该层权值W,  j为2时更新该层偏置b
					*data_[lname][j] = (*data_[lname][j]) + (*dx);   //梯度下降原则  W := W - LearningRate*dW   ；  b := b - LearningRate*db
				}
			}

			// step3.  evaluate  接下来，我们开始评估训练情况
			bool first_it = (iter == 0); //是否处于第一个迭代周期的标志
			bool epoch_end = (iter + 1) % iter_per_epochs == 0;//是否处于epoch最后一个迭代周期的标志
			bool acc_check = (param.acc_frequence && (iter + 1) % param.acc_frequence == 0);//
			if (first_it || epoch_end || acc_check) //三个条件符合一个就执行准确率测试的代码！
			{   //第一个iter默认做一次测试，最后一个iter也默认做一次测试，整个epoch中间是否测试要看网络参数设置json文件。

				// update learning rate[TODO]  学习率更新（还没实现，可以自己实现以下！）
				if ((iter > 0 && epoch_end) || param.acc_update_lr)   //或运算？那就是最后一个iter一定会进入这个if语句了！
				{//lr_decay在json文件中并未明确指明，那默认不是0吗？这样学习率不就更新为0了？要具体执行看看才行！（lr_decay默认为1才行啊）
					//std::cout<<"param.lr_decay  = "<< param.lr_decay <<std::endl;  //debug: gordon
					param.lr *= param.lr_decay;   //更新规则    【学习率 := 学习率 x 学习率衰减系数（>0或<0）】
					//std::cout<<"param.lr  = "<< param.lr <<std::endl;  //debug: gordon
					if (epoch_end)
					{
						epoch++;  //epoch自加！
					}
				}

				// evaluate train set accuracy 评估训练好的模型在训练集上的准确率！
				shared_ptr<Blob> X_train_subset;  //训练集（feature）片段
				shared_ptr<Blob> Y_train_subset;  //训练集（label）片段
				if (N > 1000)//训练集样本总数大于1000，只取一部分进行测试，以减少测试时间！
				{
					X_train_subset.reset(new Blob(X_train_->subBlob(0, 100)));  //取训练集样本前面100个作为一个片段（参与测试）
					Y_train_subset.reset(new Blob(Y_train_->subBlob(0, 100)));
				}
				else //训练集样本总数小于1000（测试时间花费不多）
				{
					X_train_subset = X_train_; //直接以全部样本参与做测试，不取片段
					Y_train_subset = Y_train_;
				}
				//再次调用trainNet函数，注意这次多加了一个mode=forward参数，只做forward（只测试，不训练，无需backward）
				trainNet(X_train_subset, Y_train_subset, param, "forward");
				//计算训练集片段上的准确率
				double train_acc = prob(*data_[layers_.back()][1], *data_[layers_.back()][0]);//参数：最末层（softmax）标签Blob；最末层输入Blob
				train_acc_history_.push_back(train_acc);  //将准确率叠加进vector，便于打印输出和描点绘图

				// evaluate val set accuracy[TODO: change train to val]
				//在整个验证集（所有样本）上做测试，获得验证集准确率
				trainNet(X_val_, Y_val_, param, "forward"); //同样是做一个前向计算（无backward）
				double val_acc = prob(*data_[layers_.back()][1], *data_[layers_.back()][0]);//参数：最末层（softmax）标签Blob；最末层输入Blob
				val_acc_history_.push_back(val_acc);//将准确率叠加进vector，便于打印输出和描点绘图

				// print 打印输出！！！
				printf("iter: %d  loss: %f  train_acc: %0.2f%%    val_acc: %0.2f%%    lr: %0.6f\n",
					iter, loss_, train_acc * 100, val_acc * 100, param.lr);

				// save best model[TODO]  保存模型快照？？作者为什么把这个屏蔽掉？ 自己模拟caffe实现快照保存也不难！
				//if (val_acc_history_.size() > 1 && val_acc < val_acc_history_[val_acc_history_.size()-2]) //这个if语句搞不懂！不管了！自己实现就行
				//{  
				//    for (auto i : layers_)
				//	{
				//        if (!data_[i][1] || !data_[i][2]) //若有一个为空则中断本轮循环
				//		{  
				//            continue;
				//        }
				//        best_model_[i][1] = data_[i][1];  //存储当前迭代周期训练得到的权值参数
				//        best_model_[i][2] = data_[i][2];  //存储当前迭代周期训练得到的bias参数
				//    }
				//}

				//在选用模型快照保存时，每snapshot_interval个周期就保存一个快照（不要iter0的）
				if (param.snap_shot && iter % param.snapshot_interval == 0 && iter>0)    
				{
					char outputFile[40];
					sprintf(outputFile, "./iter%d.gordonmodel", iter);
					//检查输出文件是否存在，打印对应信息提示我们将要自动创建 （可要可不要）
				    fstream input(outputFile, ios::in | ios::binary);
					if (!input) 
						cout << outputFile << "  was not found.  Creating a new file now." << endl;
					//创建参数存储字典
					PromptForParam(net_param.add_dict());
					fstream output(outputFile, ios::out | ios::trunc | ios::binary);  //首先寻找输出文件，不存在就创建！
					if (!net_param.SerializeToOstream(&output))
					{
						cerr << "Failed to write NetParameter." << endl;
						return;
					}
					google::protobuf::ShutdownProtobufLibrary();
				}


			}
		}//结束在单个mini-batch上的训练、参数更新、模型评估；继续迭代下一个mini-batch！
		cout << "--------------- Train done --------------" << endl;
		return;
	}//结束训练

	////-------------（利用夹逼准则）单层的梯度检验！！！作者太6了，连这个都实现了！！！-------------------
	//void Net::testLayer(NetParam& param, int lnum)  //参数：网络参数；需要梯度检验的层索引
	//{
	//	std::string ltype = ltype_[lnum];
	//	std::string lname = layers_[lnum];
	//	if (ltype == "Fc")
	//		_test_fc_layer(data_[lname], grads_[lname], grads_[layers_[lnum + 1]][0]);
	//	if (ltype == "Conv")
	//		_test_conv_layer(data_[lname], grads_[lname], grads_[layers_[lnum + 1]][0], param.params[lname]);
	//	if (ltype == "Pool")
	//		_test_pool_layer(data_[lname], grads_[lname], grads_[layers_[lnum + 1]][0], param.params[lname]);
	//	if (ltype == "Relu")
	//		_test_relu_layer(data_[lname], grads_[lname], grads_[layers_[lnum + 1]][0]);
	//	if (ltype == "Dropout")
	//		_test_dropout_layer(data_[lname], grads_[lname], grads_[layers_[lnum + 1]][0], param.params[lname]);
	//	if (ltype == "SVM")
	//		_test_svm_layer(data_[lname], grads_[lname][0]);
	//	if (ltype == "Softmax")
	//		_test_softmax_layer(data_[lname], grads_[lname][0]);
	//}

	//void Net::_test_fc_layer(vector<shared_ptr<Blob>>& in,
	//	vector<shared_ptr<Blob>>& grads,
	//	shared_ptr<Blob>& dout) {

	//	auto nfunc = [in](shared_ptr<Blob>& e) {return AffineLayer::forward(in, e); };
	//	Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc); //利用夹逼准则求导数，从而实现梯度检验！具体可见Ng的课程，讲得非常详细！
	//	Blob num_dw = Test::calcNumGradientBlob(in[1], dout, nfunc);
	//	Blob num_db = Test::calcNumGradientBlob(in[2], dout, nfunc);

	//	cout << "Test Affine Layer:" << endl;
	//	cout << "Test num_dx and dX Layer:" << endl;
	//	cout << Test::relError(num_dx, *grads[0]) << endl;
	//	cout << "Test num_dw and dW Layer:" << endl;
	//	cout << Test::relError(num_dw, *grads[1]) << endl;
	//	cout << "Test num_db and db Layer:" << endl;
	//	cout << Test::relError(num_db, *grads[2]) << endl;

	//	return;
	//}

	//void Net::_test_conv_layer(vector<shared_ptr<Blob>>& in,
	//	vector<shared_ptr<Blob>>& grads,
	//	shared_ptr<Blob>& dout,
	//	Param& param)  {

	//	auto nfunc = [in, &param](shared_ptr<Blob>& e) {return ConvLayer::forward(in, e, param); };
	//	Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
	//	Blob num_dw = Test::calcNumGradientBlob(in[1], dout, nfunc);
	//	Blob num_db = Test::calcNumGradientBlob(in[2], dout, nfunc);

	//	cout << "Test Conv Layer:" << endl;
	//	cout << "Test num_dx and dX Layer:" << endl;
	//	cout << Test::relError(num_dx, *grads[0]) << endl;
	//	cout << "Test num_dw and dW Layer:" << endl;
	//	cout << Test::relError(num_dw, *grads[1]) << endl;
	//	cout << "Test num_db and db Layer:" << endl;
	//	cout << Test::relError(num_db, *grads[2]) << endl;

	//	return;
	//}
	//void Net::_test_pool_layer(vector<shared_ptr<Blob>>& in,
	//	vector<shared_ptr<Blob>>& grads,
	//	shared_ptr<Blob>& dout,
	//	Param& param) {
	//	auto nfunc = [in, &param](shared_ptr<Blob>& e) {return PoolLayer::forward(in, e, param); };

	//	Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);

	//	cout << "Test Pool Layer:" << endl;
	//	cout << Test::relError(num_dx, *grads[0]) << endl;

	//	return;
	//}

	//void Net::_test_relu_layer(vector<shared_ptr<Blob>>& in,
	//	vector<shared_ptr<Blob>>& grads,
	//	shared_ptr<Blob>& dout) {
	//	auto nfunc = [in](shared_ptr<Blob>& e) {return ReluLayer::forward(in, e); };
	//	Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);

	//	cout << "Test ReLU Layer:" << endl;
	//	cout << Test::relError(num_dx, *grads[0]) << endl;

	//	return;
	//}

	//void Net::_test_dropout_layer(vector<shared_ptr<Blob>>& in,
	//	vector<shared_ptr<Blob>>& grads,
	//	shared_ptr<Blob>& dout,
	//	Param& param) {
	//	shared_ptr<Blob> dummy_out;
	//	auto nfunc = [in, &param](shared_ptr<Blob>& e) {return DropoutLayer::forward(in, e, param); };

	//	cout << "Test Dropout Layer:" << endl;
	//	Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
	//	cout << Test::relError(num_dx, *grads[0]) << endl;

	//	return;
	//}

	//void Net::_test_svm_layer(vector<shared_ptr<Blob>>& in,
	//	shared_ptr<Blob>& dout) {
	//	shared_ptr<Blob> dummy_out;
	//	auto nfunc = [in, &dummy_out](double& e) {return SVMLossLayer::go(in, e, dummy_out, 1); };
	//	cout << "Test SVM Loss Layer:" << endl;

	//	Blob num_dx = Test::calcNumGradientBlobLoss(in[0], nfunc);
	//	cout << Test::relError(num_dx, *dout) << endl;

	//	return;
	//}

	//void Net::_test_softmax_layer(vector<shared_ptr<Blob>>& in,
	//	shared_ptr<Blob>& dout) {
	//	shared_ptr<Blob> dummy_out;
	//	auto nfunc = [in, &dummy_out](double& e) {return SoftmaxLossLayer::go(in, e, dummy_out, 1); };

	//	cout << "Test Softmax Loss Layer:" << endl;
	//	Blob num_dx = Test::calcNumGradientBlobLoss(in[0], nfunc);
	//	cout << Test::relError(num_dx, *dout) << endl;

	//	return;
	//}









