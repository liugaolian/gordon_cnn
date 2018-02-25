#include "myLayer.hpp"

//增加几个字

/*!
* \brief forward
*             X:        [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             out:      [N, F, 1, 1]
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
* \param[in]  const Param* param            params
* \param[out] Blob& out                     Y
*/
void AffineLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out)
{
	if (out) 
	{
		out.reset();
	}
	int N = in[0]->get_N();
	int F = in[1]->get_N();

	mat x = in[0]->reshape();   //转成一维向量  reshape [N,C,H,W] to [N,C*H*W]
	mat w = in[1]->reshape();  //转成一维向量
	mat b = in[2]->reshape();  //转成一维向量
	b = repmat(b, 1, N).t();
	mat ans = x * w.t() + b;       //输出 = 输入图片 * 权值矩阵的转置  + 偏置项
	mat2Blob(ans, out, F, 1, 1);   //因为中间转成一维向量做运算，现在得转回Blob再输出！

	return;
}

/*!
* \brief backward
*             in:       [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             dout:     [N, F, 1, 1]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X, cache[1]:weights, cache[2]:bias
* \param[out] vector<Blob*>& grads          grads[0]:dX, grads[1]:dW, grads[2]:db
*/
void AffineLayer::backward(shared_ptr<Blob>& dout,    //softmax得到的dout
	const vector<shared_ptr<Blob>>& cache,  //A_prev，W, b
	vector<shared_ptr<Blob>>& grads)   //dA，dW, db
{
	shared_ptr<Blob> dX;
	shared_ptr<Blob> dW;
	shared_ptr<Blob> db;

	int n = dout->get_N();

	shared_ptr<Blob> pX = cache[0];
	shared_ptr<Blob> pW = cache[1];
	shared_ptr<Blob> pb = cache[2];

	// calc grads
	// dX  损失梯度
	mat mat_dx = dout->reshape() * pW->reshape();  //通过链式法则容易求得全连接层损失梯度
	mat2Blob(mat_dx, dX, pX->size());  //二维mat转为四维Blob
	grads[0] = dX;
	// dW 权重梯度
	mat mat_dw = dout->reshape().t() * pX->reshape();   //通过链式法则容易求得全连接层权值梯度
	mat2Blob(mat_dw, dW, (*pW).size());//二维mat转为四维Blob
	grads[1] = dW;
	// db 偏置梯度
	mat mat_db = dout->reshape().t() * mat(n, 1, fill::ones);   //通过链式法则容易求得全连接层bias梯度
	mat2Blob(mat_db, db, (*pb).size());//二维mat转为四维Blob
	grads[2] = db;

	return;
}

/*!
* \brief convolutional layer forward
*             X:        [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             out:      [N, F, (Hx+pad*2-Hw)/stride+1, (Wx+pad*2-Ww)/stride+1]
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
* \param[in]  const ConvParam* param        conv params: stride, pad
* \param[out] Blob** out                    Y
*/
void ConvLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, Param& param)
{
	if (out)
	{
		out.reset();   //清空输出Blob！
	}
	assert(in[0]->get_C() == in[1]->get_C());  //断言：图片通道数和卷积核通道数一样（务必保证这一点）
	int N = in[0]->get_N();       //图片个数
	int F = in[1]->get_N();		//卷积核个数
	int C = in[0]->get_C();        //图片通道数
	int Hx = in[0]->get_H();     //卷积前图片高度
	int Wx = in[0]->get_W();   //卷积前图片宽度
	int Hw = in[1]->get_H();    //卷积核高度
	int Ww = in[1]->get_W();  //卷积核宽度

	// calc Hy, Wy
	int Hy = (Hx + param.conv_pad * 2 - Hw) / param.conv_stride + 1;    //卷积后图片高度（计算公式画个图就懂了）
	int Wy = (Wx + param.conv_pad * 2 - Ww) / param.conv_stride + 1;  //卷积后图片宽度

	out.reset(new Blob(N, F, Hy, Wy));
	Blob padX = (*in[0]).pad(param.conv_pad);  //padding操作，得到padding后的图片（参与卷积）
	//开始卷积！
	for (int n = 0; n < N; ++n)
	{
		for (int f = 0; f < F; ++f)
		{
			for (int hh = 0; hh < Hy; ++hh)
			{
				for (int ww = 0; ww < Wy; ++ww)
				{
					cube window = padX[n](span(hh * param.conv_stride, hh * param.conv_stride + Hw - 1),
						span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
						span::all);
					(*out)[n](hh, ww, f) = accu(window % (*in[1])[f]) + as_scalar((*in[2])[f]);
				}
			}
		}
	}
	return;
}

/*!
* \brief backward
*             in:       [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             out:      [N, F, (Hx+pad*2-Hw)/stride+1, (Wx+pad*2-Ww)/stride+1]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X, cache[1]:weights, cache[2]:bias
* \param[out] vector<Blob*>& grads          grads[0]:dX, grads[1]:dW, grads[2]:db
*/
void ConvLayer::backward(shared_ptr<Blob>& dout,//输入参数：Relu层得到的损失梯度（上一层传过来的误差信号） 
	const vector<shared_ptr<Blob>>& cache, //-- conv层A_prev，W, b  
	vector<shared_ptr<Blob>>& grads, //--conv层反向计算得到的dA，dW, db 
	Param& param)  //-- conv层kernel宽高和步长参数
{
	int N = cache[0]->get_N(); //输入Blob的cube个数（样本数）
	int F = cache[1]->get_N(); //卷积核个数
	int C = cache[0]->get_C();  //输入Blob的深度（也即是cube深度）
	int Hx = cache[0]->get_H();
	int Wx = cache[0]->get_W();
	int Hw = cache[1]->get_H();
	int Ww = cache[1]->get_W();
	int Hy = dout->get_H();//损失梯度Blob的高
	int Wy = dout->get_W();//损失梯度Blob的宽
	assert(C == cache[1]->get_C());
	assert(F == cache[2]->get_N());

	//创建存储梯度的Blob变量，维数与该层输入Blob是一模一样的！
	shared_ptr<Blob> dX(new Blob(cache[0]->size(), TZEROS));  //初始化为0
	shared_ptr<Blob> dW(new Blob(cache[1]->size(), TZEROS));
	shared_ptr<Blob> db(new Blob(cache[2]->size(), TZEROS));

	//如果前向计算时对输入Blob做了padding，那么反向计算梯度时也得将存储梯度的Blob做一下padding！这样维数才匹配！
	Blob pad_dX(N, C, Hx + param.conv_pad * 2, Wx + param.conv_pad * 2, TZEROS);
	Blob pad_X = (*cache[0]).pad(1);//对输入Blob做pad=1的操作（默认填充0）

	for (int n = 0; n < N; ++n)//输入Blob的cube个数（样本数）
	{
		for (int f = 0; f < F; ++f)//卷积核个数
		{
			for (int hh = 0; hh < Hy; ++hh) //损失梯度Blob的高
			{
				for (int ww = 0; ww < Wy; ++ww) //损失梯度Blob的宽
				{
					cube window = pad_X[n](span(hh * param.conv_stride, hh * param.conv_stride + Hw - 1),
						span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
						span::all);
					(*db)[f](0, 0, 0) += (*dout)[n](hh, ww, f);
					(*dW)[f] += window * (*dout)[n](hh, ww, f);
					pad_dX[n](span(hh * param.conv_stride, hh * param.conv_stride + Hw - 1),
						span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
						span::all) += (*cache[1])[f] * (*dout)[n](hh, ww, f);  //自加操作！
				}
			}
		}
	}
	*dX = pad_dX.dePad(param.conv_pad);  //去除padding部分
	grads[0] = dX;  //将本层损失梯度存好
	grads[1] = dW;//将本层权值梯度存好
	grads[2] = db;//将本层bias梯度存好

	return;
}

/*!
* \brief forward
*             X:        [N, C, Hx, Wx]
*             out:      [N, C, Hx/2, Wx/2]
* \param[in]  const vector<Blob*>& in       in[0]:X
* \param[in]  const Param* param        conv params
* \param[out] Blob& out                     Y
*/
void PoolLayer::forward(const vector<shared_ptr<Blob>>& in,
	shared_ptr<Blob>& out,
	Param& param) {
	if (out) {
		out.reset();
	}
	int N = (*in[0]).get_N();
	int C = (*in[0]).get_C();
	int Hx = (*in[0]).get_H();
	int Wx = (*in[0]).get_W();
	int height = param.pool_height;
	int width = param.pool_width;
	int stride = param.pool_stride;

	int Hy = (Hx - height) / stride + 1;
	int Wy = (Wx - width) / stride + 1;

	out.reset(new Blob(N, C, Hy, Wy));

	for (int n = 0; n < N; ++n) {
		for (int c = 0; c < C; ++c) {
			for (int hh = 0; hh < Hy; ++hh) {
				for (int ww = 0; ww < Wy; ++ww) {
					(*out)[n](hh, ww, c) = (*in[0])[n](span(hh * stride, hh * stride + height - 1),
						span(ww * stride, ww * stride + width - 1),
						span(c, c)).max();
				}
			}
		}
	}
	return;
}

/*!
* \brief backward
*             cache:    [N, C, Hx, Wx]
*             dout:     [N, F, Hx/2, Wx/2]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X
* \param[out] vector<Blob*>& grads          grads[0]:dX
*/
void PoolLayer::backward(shared_ptr<Blob>& dout,				//输入参数：softmax得到的dout
	const vector<shared_ptr<Blob>>& cache,	//全连接层A_prev，W, b
	vector<shared_ptr<Blob>>& grads,				//全连接层dA，dW, db
	Param& param)   //池化层的尺寸（H，W），步长等参数
{
	int N = cache[0]->get_N();
	int C = cache[0]->get_C();
	int Hx = cache[0]->get_H();
	int Wx = cache[0]->get_W();
	int Hy = dout->get_H();
	int Wy = dout->get_W();
	int height = param.pool_height;
	int width = param.pool_width;
	int stride = param.pool_stride;

	shared_ptr<Blob> dX(new Blob(cache[0]->size(), TZEROS));  //该层损失梯度Blob维数和输入Blob维数相同

	for (int n = 0; n < N; ++n)
	{
		for (int c = 0; c < C; ++c)
		{
			for (int hh = 0; hh < Hy; ++hh)
			{
				for (int ww = 0; ww < Wy; ++ww)
				{
					mat window = (*cache[0])[n](span(hh * stride, hh * stride + height - 1),
						span(ww * stride, ww * stride + width - 1),
						span(c, c));
					double maxv = window.max();
					mat mask = conv_to<mat>::from(maxv == window);  //掩码操作！
					(*dX)[n](span(hh * stride, hh * stride + height - 1),
						span(ww * stride, ww * stride + width - 1),
						span(c, c)) += mask * (*dout)[n](hh, ww, c);
				}
			}
		}
	}
	grads[0] = dX;   //注意：池化层只有损失梯度，没有权重梯度和bias梯度
	return;
}

/*!
* \brief forward, out = max(0, X)
*             X:        [N, C, Hx, Wx]
*             out:      [N, C, Hx, Wx]
* \param[in]  const vector<Blob*>& in       in[0]:X
* \param[out] Blob& out                     Y
*/
void ReluLayer::forward(const vector<shared_ptr<Blob>>& in,
	shared_ptr<Blob>& out) {
	if (out) {
		out.reset();
	}
	out.reset(new Blob(*in[0]));
	(*out).maxIn(0);
	return;
}

/*!
* \brief backward, dX = dout .* (X > 0)
*             in:       [N, C, Hx, Wx]
*             dout:     [N, F, Hx, Wx]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X
* \param[out] vector<Blob*>& grads          grads[0]:dX
*/
void ReluLayer::backward(shared_ptr<Blob>& dout,//pool层得到的损失梯度 （误差信号）
	const vector<shared_ptr<Blob>>& cache, //-- Relu层A_prev，W, b 
	vector<shared_ptr<Blob>>& grads)  //-- Relu层反向计算得到的dA，dW, db
{
	shared_ptr<Blob> dX(new Blob(*cache[0]));
	int N = cache[0]->get_N();  //输入Blob的cube个数（样本个数）
	for (int i = 0; i < N; ++i)   //遍历cube（样本）
	{
		(*dX)[i].transform([](double e) {return e > 0 ? 1 : 0; });   //ReLU的梯度计算很简单：输入大于0则梯度为1，输入小于0则梯度为0
	}
	(*dX) = (*dout) * (*dX);  //计算当前层的损失梯度（误差信号）
	grads[0] = dX; //注意：Relu层也是只有损失梯度，没有权重梯度和bias梯度
	return;
}

/*!
* \brief forward
*             X:        [N, C, Hx, Wx]
*             out:      [N, C, Hx, Wx]
* \param[in]  const vector<Blob*>& in       in[0]:X
* \param[in]  Param& param                  int mode, double p, int seed, Blob *mask
* \param[out] Blob& out                     Y
*/
void DropoutLayer::forward(const vector<shared_ptr<Blob>>& in,
	shared_ptr<Blob>& out,
	Param& param) {
	if (out) {
		out.reset();
	}
	int mode = param.drop_mode;
	double p = param.drop_p;
	assert(0 <= p && p <= 1);
	assert(0 <= mode && mode <= 3);
	int seed;
	/*! train mode */
	if ((mode & 1) == 1) {
		if ((mode & 2) == 2) {
			seed = param.drop_seed;
		}
		shared_ptr<Blob> mask(new Blob(seed, in[0]->size(), TRANDU));
		(*mask).smallerIn(p);
		Blob in_mask = (*in[0]) * (*mask);
		out.reset(new Blob(in_mask / p));
		if (param.drop_mask) {
			param.drop_mask.reset();
		}
		param.drop_mask = mask;
	}
	else {
		/*! test mode */
		out.reset(new Blob(*in[0]));
	}
	return;
}

/*!
* \brief backward
*             in:       [N, C, Hx, Wx]
*             dout:     [N, F, Hx, Wx]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X
* \param[in]  Param& param                  int mode, double p, int seed, Blob *mask
* \param[out] vector<Blob*>& grads          grads[0]:dX
*/
void DropoutLayer::backward(shared_ptr<Blob>& dout,
	const vector<shared_ptr<Blob>>& cache,
	vector<shared_ptr<Blob>>& grads,
	Param& param) {
	shared_ptr<Blob> dX(new Blob((*dout)));
	int mode = param.drop_mode;
	assert(0 <= mode && mode <= 3);
	if ((mode & 1) == 1) {
		Blob dx_mask = (*dX) * (*param.drop_mask);
		*dX = dx_mask / param.drop_p;
	}
	grads[0] = dX;
	return;
}

/*!
* \brief forward
*             X:        [N, C, 1, 1], usually the output of affine(fc) layer
*             Y:        [N, C, 1, 1], ground truth, with 1(true) or 0(false)
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:Y
* \param[out] double& loss                  loss
* \param[out] Blob** out                    out: dX
*/
void SoftmaxLossLayer::go(const vector<shared_ptr<Blob>>& in,
	double& loss,
	shared_ptr<Blob>& dout,
	int mode)  //mode控制用于训练，还是用于测试！
{
	//Blob X(*in[0]);
	//Blob Y(*in[1]);
	if (dout) //不为空
	{
		dout.reset(); //清空
	}
	//获取输入Blob的参数
	int N = in[0]->get_N(); //cube个数
	int C = in[0]->get_C();
	int H = in[0]->get_H();
	int W = in[0]->get_W();
	assert(H == 1 && W == 1); //断言：输入Blob的宽高都为1（它是由全连接层输出得到的，所以合成为Blob时，维数都体现在C里面了）

	mat mat_x = in[0]->reshape();  //将输入Blob变为二维矩阵，即：reshape [N,C,H,W] to [N,C*H*W]
	mat mat_y = in[1]->reshape();  //将标签Blob变为二维矩阵

	/*! forward */
	mat row_max = repmat(arma::max(mat_x, 1), 1, C);
	mat_x = arma::exp(mat_x - row_max);
	mat row_sum = repmat(arma::sum(mat_x, 1), 1, C);
	mat e = mat_x / row_sum;
	//e.print("e:\n");
	//mat rrs = arma::sum(e, 1);
	//rrs.print("rrs:\n");
	mat prob = -arma::log(e);
	//prob.print("prob:\n");
	//(prob%mat_y).print("gg:\n");
	/*! loss should near -log(1/C) */
	loss = accu(prob % mat_y) / N;
	/*! only forward 仅仅前向传播！即用于测试阶段，不是训练阶段*/
	if (mode == 1)   //mode控制用于训练，还是用于测试！
		return;

	/*! backward 计算损失梯度！*/
	mat dx = e - mat_y;
	dx /= N;    //损失梯度计算，注意要除以cube个数（样本个数），具体查看Ng课程公式就知道
	mat2Blob(dx, dout, (*in[0]).size());  //dx以Blob格式存入dout，因为层与层之间数据传播，都是以Blob为基本格式传递的！所以要将mat转换回Blob
	return;
}

/*!
* \brief forward
*             X:        [N, C, 1, 1], usually the output of affine(fc) layer
*             Y:        [N, C, 1, 1], ground truth, with 1(true) or 0(false)
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:Y
* \param[out] double& loss                  loss
* \param[out] Blob** out                    out: dX
* \param[in]  int mode                      1: only forward, 0:forward and backward
*/
void SVMLossLayer::go(const vector<shared_ptr<Blob>>& in,
	double& loss,
	shared_ptr<Blob>& dout,
	int mode) {
	if (dout) {
		dout.reset();
	}
	/*! let delta equals to 1 */
	double delta = 0.2;
	int N = in[0]->get_N();
	int C = in[0]->get_C();
	mat mat_x = in[0]->reshape();
	mat mat_y = in[1]->reshape();
	//mat_x.print("X:\n");
	//mat_y.print("Y:\n");

	/*! forward */
	mat good_x = repmat(arma::sum(mat_x % mat_y, 1), 1, C);
	mat mat_loss = (mat_x - good_x + delta);
	mat_loss.transform([](double e) {return e > 0 ? e : 0; });
	mat_y.transform([](double e) {return e ? 0 : 1; });
	mat_loss %= mat_y;
	loss = accu(mat_loss) / N;
	if (mode == 1)
		return;

	/*! backward */
	mat dx(mat_loss);
	dx.transform([](double e) {return e ? 1 : 0; });
	mat_y.transform([](double e) {return e ? 0 : 1; });
	mat sum_x = repmat(arma::sum(dx, 1), 1, C) % mat_y;
	dx = (dx - sum_x) / N;
	mat2Blob(dx, dout, in[0]->size());
	return;
}