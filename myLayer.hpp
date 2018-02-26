#ifndef  __MYLAYER_HPP__
#define __MYLAYER_HPP__

#include "myBlob.hpp"
#include <memory>
using std::vector;
using std::shared_ptr;
/*! layer parameters */
struct Param 
{
	Param() : conv_stride(0), conv_pad(0)
	{}
	/*! \brief conv param */
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;
	inline void setConvParam(int stride, int pad, int width, int height, int kernels) 
	{
		conv_stride = stride;
		conv_pad = pad;
		conv_width = width;
		conv_height = height;
		conv_kernels = kernels;
	}
	/*! \brief pool param */
	int pool_stride;
	int pool_width;
	int pool_height;
	inline void setPoolParam(int stride, int width, int height) 
	{
		pool_stride = stride;
		pool_width = width;
		pool_height = height;
	}
	/*! \brief dropout param */
	/*! if the most right bit is 1 use train mode, else use test mode;
	*  if the second bit from right is 1, use random seed; else use selected seed. */
	int drop_mode;
	double drop_p;
	int drop_seed;
	shared_ptr<Blob> drop_mask;
	inline void setDropoutpParam(int mode, double pp, int s)
	{
		drop_mode = mode;
		drop_p = pp;
		drop_seed = s;
		drop_mask.reset();
	}
	/*! fc parameters */
	int fc_kernels;
};

/*!
* \brief Affine Layer
*/
class AffineLayer   //全连接层
{
public:
	AffineLayer() {}
	~AffineLayer() {}

	/*!
	* \brief forward
	* Blob bottom[0]:                                                                 in[1]:weight
	*     _______          _______         __  _______________________         __      __  _______________________          __
	*  C /______/|   N    /______/|        |  |_______________________| __      |      |  |_______________________| __       |
	*   |------||| ······|------|||   ===> |          ...                |      |   *  |            ...              |       | . T() + b
	* H |------|||       |------|||   ===> |   _______________________    > N   |      |   _______________________    > F    |
	*   |------|/        |------|/         |_ |_______________________| _|     _|      |_ |_______________________| _|      _|
	*      W
	*   \___________  __________/             \___________  __________/                  \___________  __________/
	*               \/                                    \/                                         \/
	*           [N,C,H,W]                               C*H*W                                       C*H*N
	*
	*             X:        [N, C, Hx, Wx]
	*             weight:   [F, C, Hw, Ww]
	*             bias:     [F, 1, 1, 1]
	*             out:      [N, F, 1, 1]
	* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
	* \param[out] Blob& out                     Y
	*/
	static void forward(const vector<shared_ptr<Blob>>& in,
		shared_ptr<Blob>& out);

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
	static void backward(shared_ptr<Blob>& dout,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads);
};

/*!
* \brief Convolutional Layer
*/
class ConvLayer {   //卷积层
public:
	ConvLayer() {}
	~ConvLayer() {}

	/*!
	* \brief forward
	*  Blob bottom[0]:
	*     _______          _______                                               _______          _______
	*  C /______/|   N    /______/|                                           C /______/|   N*F  /______/|
	*   |------||| ······|------|||                                            |------||| ······|------|||
	* H |------|||       |------|||    *   F个kernel size为(n,n)的卷积核  =  Hw |------|||       |------||| Hw
	*   |------|/        |------|/                                             |------|/        |------|/
	*      W                                                                      Ww               Ww
	*   \___________  __________/
	*               \/
	*            [N,C,H,W]
	*
	*             X:        [N, C, Hx, Wx]
	*             weight:   [F, C, Hw, Ww]
	*             bias:     [F, 1, 1, 1]
	*             out:      [N, F, (Hx+pad*2-Hw)/stride+1, (Wx+pad*2-Ww)/stride+1]
	* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
	* \param[in]  const ConvParam* param        conv params
	* \param[out] Blob& out                     Y
	*/
	static void forward(const vector<shared_ptr<Blob>>& in,
		shared_ptr<Blob>& out,
		Param& param);

	/*!
	* \brief backward
	*             in:       [N, C, Hx, Wx]
	*             weight:   [F, C, Hw, Ww]
	*             bias:     [F, 1, 1, 1]
	*             dout:     [N, F, (Hx+pad*2-Hw)/stride+1, (Wx+pad*2-Ww)/stride+1]
	* \param[in]  const Blob* dout              dout
	* \param[in]  const vector<Blob*>& cache    cache[0]:X, cache[1]:weights, cache[2]:bias
	* \param[out] vector<Blob*>& grads          grads[0]:dX, grads[1]:dW, grads[2]:db
	*/
	static void backward(shared_ptr<Blob>& dout,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads,
		Param& param);
};

/*!
* \brief Max Pooling Layer
*/
class PoolLayer   //池化层
{
public:
	PoolLayer() {}
	~PoolLayer() {}

	/*!
	* \brief forward
	*             X:        [N, C, Hx, Wx]
	*             out:      [N, C, Hx/2, Wx/2]
	* \param[in]  const vector<Blob*>& in       in[0]:X
	* \param[in]  const Param* param        conv params
	* \param[out] Blob& out                     Y
	*/
	static void forward(const vector<shared_ptr<Blob>>& in,
		shared_ptr<Blob>& out,
		Param& param);

	/*!
	* \brief backward
	*             in:       [N, C, Hx, Wx]
	*             dout:     [N, F, Hx/2, Wx/2]
	* \param[in]  const Blob* dout              dout
	* \param[in]  const vector<Blob*>& cache    cache[0]:X
	* \param[out] vector<Blob*>& grads          grads[0]:dX
	*/
	static void backward(shared_ptr<Blob>& dout,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads,
		Param& param);
};

/*!
* \brief ReLU Layer
*/
class ReluLayer   //ReLU激活函数层
{
public:
	ReluLayer() {}
	~ReluLayer() {}

	/*!
	* \brief forward, out = max(0, X)
	*             X:        [N, C, Hx, Wx]
	*             out:      [N, C, Hx, Wx]
	* \param[in]  const vector<Blob*>& in       in[0]:X
	* \param[out] Blob& out                     Y
	*/
	static void forward(const vector<shared_ptr<Blob>>& in,
		shared_ptr<Blob>& out);

	/*!
	* \brief backward, dX = dout .* (X > 0)
	*             in:       [N, C, Hx, Wx]
	*             dout:     [N, F, Hx, Wx]
	* \param[in]  const Blob* dout              dout
	* \param[in]  const vector<Blob*>& cache    cache[0]:X
	* \param[out] vector<Blob*>& grads          grads[0]:dX
	*/
	static void backward(shared_ptr<Blob>& dout,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads);
};

/*!
* \brief Dropout Layer
*/
class DropoutLayer   //Dropout 层
{
public:
	DropoutLayer() {}
	~DropoutLayer() {}

	/*!
	* \brief forward
	*             X:        [N, C, Hx, Wx]
	*             out:      [N, C, Hx, Wx]
	* \param[in]  const vector<Blob*>& in       in[0]:X
	* \param[out] Blob& out                     Y
	*/
	static void forward(const vector<shared_ptr<Blob>>& in,
		shared_ptr<Blob>& out,
		Param& param);

	/*!
	* \brief backward
	*             in:       [N, C, Hx, Wx]
	*             dout:     [N, F, Hx, Wx]
	* \param[in]  const Blob* dout              dout
	* \param[in]  const vector<Blob*>& cache    cache[0]:X
	* \param[out] vector<Blob*>& grads          grads[0]:dX
	*/
	static void backward(shared_ptr<Blob>& dout,
		const vector<shared_ptr<Blob>>& cache,
		vector<shared_ptr<Blob>>& grads,
		Param& param);
};

/*!
* \brief Softmax Loss Layer
*/
class SoftmaxLossLayer       //损失函数层
{
public:
	SoftmaxLossLayer() {}
	~SoftmaxLossLayer() {}

	/*!
	* \brief forward
	*             X:        [N, C, 1, 1], usually the output of affine(fc) layer
	*             Y:        [N, C, 1, 1], ground truth, with 1(true) or 0(false)
	* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:Y
	* \param[out] double& loss                  loss
	* \param[out] Blob** out                    out: dX
	* \param[in]  int mode                      1: only forward, 0:forward and backward
	*/
	static void go(const vector<shared_ptr<Blob>>& in,
		double& loss,
		shared_ptr<Blob>& dout,
		int mode = 0);
};

/*!
* \brief SVM Loss Layer
*/
class SVMLossLayer {
public:
	SVMLossLayer() {}
	~SVMLossLayer() {}

	/*!
	* \brief forward
	*             X:        [N, C, 1, 1], usually the output of affine(fc) layer
	*             Y:        [N, C, 1, 1], ground truth, with 1(true) or 0(false)
	* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:Y
	* \param[out] double& loss                  loss
	* \param[out] Blob** out                    out: dX
	* \param[in]  int mode                      1: only forward, 0:forward and backward
	*/
	static void go(const vector<shared_ptr<Blob>>& in,
		double& loss,
		shared_ptr<Blob>& dout,
		int mode = 0);
};


#endif  //__MYLAYER_HPP__


	


