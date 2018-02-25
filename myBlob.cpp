#include "myBlob.hpp"
#include <iostream>
//增加几个字

Blob operator+(Blob& A, double num)
{
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] + num;
    }
    return out;
}
Blob operator+(double num, Blob& A)
{
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] + num;
    }
    return out;
}
Blob operator+(Blob& A, Blob& B) 
{
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        assert(sz_A[i] == sz_B[i]);
    }
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] + B[i];
    }
    return out;
}

Blob operator-(Blob& A, double num)
{
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] - num;
    }
    return out;
}
Blob operator-(double num, Blob& A) 
{
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = num - A[i] ;
    }
    return out;
}
Blob operator-(Blob& A, Blob& B) 
{
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        assert(sz_A[i] == sz_B[i]);
    }
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] - B[i];
    }
    return out; 
}
Blob operator*(Blob& A, double num) 
{
    Blob out(A.size());  //
    int N = A.get_N();  //该Blob的cube数
    for (int i = 0; i < N; ++i) 
	{
        out[i] = A[i] * num;
    }
    return out;
}
Blob operator*(double num, Blob& A) 
{
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] * num;
    }
    return out;
}
Blob operator*(Blob& A, Blob& B) 
{
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        assert(sz_A[i] == sz_B[i]);
    }
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] % B[i];
    }
    return out;
}
Blob operator/(Blob& A, double num) 
{
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] / num;
    }
    return out;
}
Blob operator/(double num, Blob& A)
{
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = num / A[i];
    }
    return out;
}
Blob operator/(Blob& A, Blob& B)
{
    vector<int> sz_A = A.size();
    vector<int> sz_B = B.size();
    for (int i = 0; i < 4; ++i) {
        assert(sz_A[i] == sz_B[i]);
    }
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = A[i] / B[i];
    }
    return out;
}

Blob sqrt(Blob& A) 
{
    Blob out(A.size());
    int N = A.get_N();
    for (int i = 0; i < N; ++i) {
        out[i] = arma::sqrt(A[i]);
    }
    return out;
}

double prob(Blob& Y, Blob& p) //参数：最末层（softmax）标签Blob；最末层输入Blob
{
    assert(Y.get_N() == p.get_N());
    assert(Y.get_C() == p.get_C());
    assert(Y.get_H() == p.get_H());
    assert(Y.get_W() == p.get_W());
	//注意！！！标签Blob的格式：[N, C, 1, 1] ，对于mnist数据集来说就是[10000,10,1,1]
    double ret = 0;
    int N = Y.get_N();
    int C = Y.get_C();
	//std::cout<<"Y.get_N()  = "<< N <<std::endl;  //debug: gordon  实际在linux调试时多打印看看Blob格式，搞清楚Blob和mat是如何转换的
	//std::cout << "Y.get_C()  = " << C << std::endl;  //debug: gordon
    vector<int> pp(N, -1); //创建一个vector，有N个元素，且值均初始化为-1（这个无特殊意义，随便初始化的，避免和便签数字重合，小于0就行了）
    vector<int> yy(N, -2); //创建一个vector，有N个元素，且值均初始化为-2（这个无特殊意义，随便初始化的，避免和便签数字重合，小于0就行了）

	mat mpp = p.reshape(); //输入Blob变换为mat格式  [N, C, 1, 1] --> [N, C]
    mat myy = Y.reshape(); //标签Blob变换为mat格式   [N, C, 1, 1] --> [N, C]
    
    for (int i = 0; i < N; ++i) //遍历矩阵行（遍历样本）
	{
        int idx_p = 0, idx_y = 0;
		double max_p = mpp(i, 0);  //
		double max_y = myy(i, 0);  //

        for (int j = 1; j < C; ++j) //遍历矩阵列（遍历每个样本对应的全连接层神经元的输出值）
		{
            if (mpp(i, j) > max_p)    //从所有输入中找到最大值！
			{
                max_p = mpp(i, j);
                idx_p = j;
            }
            if (myy(i, j) > max_y)   //从所有标签中找到最大值！
			{
                max_y = myy(i, j);
                idx_y = j;
            }
        }
        pp[i] = idx_p; //将某输入（和某个样本对应）最大值所在位置存入vector
        yy[i] = idx_y;  //将某标签（和某个样本对应）最大值所在位置存入vector
    }
    int cnt = 0;
    for (int i = 0; i < N; ++i)  //遍历矩阵行（遍历样本）
	{
        if (pp[i] == yy[i])  //比较输入最大值和标签最大值，若相等则预测正确！
            cnt++;  //正确个数自加
    }
    ret = (double)cnt / (double)N;   //准确率 = 正确个数/总数
    return ret;
}

Blob compare(Blob& A, Blob& B) {
    assert(A.get_N() == B.get_N());
    Blob out(A.size());
    for (int i = 0; i < A.get_N(); ++i) {
        out[i] = conv_to<cube>::from(A[i] == B[i]);
    }
    return out;
}

// convertion
void mat2Blob(mat& mA, shared_ptr<Blob>& out, int c, int h, int w) 
{
    int n = mA.n_rows;
    assert(mA.n_cols == c*h*w);

    mA = mA.t();
    if (out) {
        out.reset();
    }
    out.reset(new Blob(n, c, h, w));
    for (int i = 0; i < n; ++i) {
        (*out)[i] = cube(mA.colptr(i), h, w, c);
    }
    return;
}

void mat2Blob(mat& mA, shared_ptr<Blob>& out, const vector<int>& sz) 
{
    int n = mA.n_rows;
    int c = sz[1];
    int h = sz[2];
    int w = sz[3];
    assert(mA.n_cols == c*h*w);

    mA = mA.t();
    if (out) {
        out.reset();
    }
    out.reset(new Blob(n, c, h, w));
    for (int i = 0; i < n; ++i) {
        (*out)[i] = cube(mA.colptr(i), h, w, c);
    }
    return;
}

// += -= *= /=
Blob& Blob::operator+=(const double num) {
    for (int i=0; i<N_; ++i) {
        data_[i] = data_[i] + num;
    }
    return *this;
}
Blob& Blob::operator-=(const double num)
{
    for (int i=0; i<N_; ++i) {
        data_[i] = data_[i] - num;
    }
    return *this;
}
Blob& Blob::operator*=(const double num) 
{
    for (int i=0; i<N_; ++i) {
        data_[i] = data_[i] * num;
    }
    return *this;
}

Blob& Blob::operator/=(const double num) 
{
    for (int i=0; i<N_; ++i) {
        data_[i] = data_[i] / num;
    }
    return *this;
}

//---Blob---
void Blob::setShape(vector<int>& shape) 
{
    N_ = shape[0];
    C_ = shape[1];
    H_ = shape[2];
    W_ = shape[3];
    data_ = vector<cube>(N_, cube(H_, W_, C_));
    return;
}

//构造函数（直接指定nchw数字创建对象）
Blob::Blob(const int n, const int c, const int h, const int w, int type) :N_(n), C_(c), H_(h), W_(w) 
{
    arma_rng::set_seed_random();
    _init(n, c, h, w, type);  //调用初始化函数完成Blob的对象的建立
    return;
}
//构造函数（输入Blob形状来构造）
Blob::Blob(const vector<int>& shape, int type) : N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3]) 
{
    arma_rng::set_seed_random();  //系统随机生成种子
    _init(N_, C_, H_, W_, type);
    return;
}

Blob::Blob(int seed, const vector<int>& shape, int type): N_(shape[0]), C_(shape[1]), H_(shape[2]), W_(shape[3])
{
    arma_rng::set_seed(seed);  //指定的随机数生成种子
    _init(N_, C_, H_, W_, type);
    return;
}

cube& Blob::operator[] (int i) {
    return data_[i];
}
Blob& Blob::operator= (double num) {
    assert(!data_.empty());
    for (int i = 0; i < N_; ++i) {
        (*this)[i].fill(num);
    }
    return *this;
}

vector<int> Blob::size() {
    vector<int> shape{ N_, C_, H_, W_ };
    return shape;   
}

vector<cube>& Blob::get_data() {
    return data_;
}

mat Blob::reshape() 
{
    cube out;
    for (int i = 0; i < N_; ++i)  //遍历前向输入Blob的所有cube（注意这里的data_是Blob类里面定义的，跟输入前向向量data_要区别开！）
	{
        out = join_slices(out, data_[i]);
    }
    //return arma::reshape(vectorise(dst), N_, C_*H_*W_);
    //return arma::reshape(vectorise(dst), C_*H_*W_, N_).t();
    out.reshape(C_*H_*W_, N_, 1);
    return out.slice(0).t(); //转置
}

double Blob::sum() {
    assert(!data_.empty());
    double ans = 0;
    for (int i = 0; i < N_; ++i) {
        ans += accu(data_[i]);
    }
    return ans;
}

double Blob::numElement() {
    return N_ * C_ * H_ * W_;
}

Blob Blob::max(double val) {
    assert(!data_.empty());
    Blob out(*this);
    for (int i = 0; i < N_; ++i) {
        out[i].transform([val](double e) {return e > val ? e : val;});
    }
    return out;
}

void Blob::maxIn(double val) {
    assert(!data_.empty());
    for (int i = 0; i < N_; ++i) {
        (*this)[i].transform([val](double e) {return e > val ? e : val;});
    }
    return;
}

/*! element wise operation, change data_, if e = e < val ? 1 : 0; */
void Blob::smallerIn(double val) {
    assert(!data_.empty());
    for (int i = 0; i < N_; ++i) {
        (*this)[i].transform([val](double e) {return e < val ? 1 : 0;});
    }
    return;
}

/*! element wise operation, change data_, if e = e > val ? 1 : 0; */
void Blob::biggerIn(double val) {
    assert(!data_.empty());
    for (int i = 0; i < N_; ++i) {
        (*this)[i].transform([val](double e) {return e > val ? 1 : 0;});
    }
    return;
}

Blob Blob::abs() {
    assert(!data_.empty());
    Blob out(*this);
    for (int i = 0; i < N_; ++i) {
        out[i].transform([](double e) {return fabs(e);});
    }
    return out;
}

/*! sub Blob, return [lo, hi) */
Blob Blob::subBlob(int lo, int hi)    //截取一段Blob当做单个批次数据
{
    if (hi < lo)
	{
        //cout << "subBlob overflow.\n" << endl;
        Blob out(hi+N_-lo, C_, H_, W_);
        for (int i = lo; i < N_; ++i) {
            out[i - lo] = (*this)[i];
        }
        for (int i = 0; i < hi; ++i) {
            out[i+N_-lo] = (*this)[i];
        }
        return out;
    }
    else 
	{
        Blob out(hi-lo, C_, H_, W_);
        for (int i = lo; i < hi; ++i) {
            out[i - lo] = (*this)[i];
        }
        return out;
    }
}

double Blob::maxVal() {
    assert(!data_.empty());
    double ans = data_[0].max();
    for (int i = 1; i < N_; ++i) {
        double tmp = data_[i].max();
        ans = std::max(ans, tmp);
    }
    return ans;
}

Blob Blob::pad(int p, double val) 
{
    assert(!data_.empty());
    Blob out(N_, C_, H_ + p*2, W_ + p*2);  //padding后的输出Blob
    out = val;   //先以padding值填充整个Blob

    for (int n = 0; n < N_; ++n)
	{
        for (int c = 0; c < C_; ++c) 
		{
            for (int h = 0; h < H_; ++h)
			{
                for (int w = 0; w < W_; ++w) 
				{
                    out[n](p+h, p+w, c) = (*this)[n](h, w, c);  //再以原图像素值填充非padding部分（做一幅画，先把背景全部刷白，再在中间作画）
                }
            }
        }
    }
    return out;
}

Blob Blob::dePad(int p)
{
    assert(!data_.empty());
    Blob out(N_, C_, H_ - p*2, W_ - p*2);

    for (int n = 0; n < N_; ++n)
	{
        //out[n] = (*this)[n](span(p, H_-p), span(p, W_-p), span::all);
        for (int c = 0; c < C_; ++c) 
		{
            for (int h = p; h < H_-p; ++h)
			{
                for (int w = p; w < W_-p; ++w) 
				{
                    out[n](h-p, w-p, c) = (*this)[n](h, w, c);
                }
            }
        }
    }
    return out;
}

void Blob::print(std::string s) {
    assert(!data_.empty());
    cout << s << endl;
    for (int i = 0; i < N_; ++i) {
        printf("N = %d\n", i);
        (*this)[i].print();
    }
    return;
}

void Blob::_init(int n, int c, int h, int w, int type) //初始化Blob函数
{
	//1.定义Blob的形状
    if (type == TONES) 
	{
		//std::cout << "init Blob:TONES" << endl;//debug:gordon
        data_ = vector<cube>(n, cube(h, w, c, fill::ones));  //定义vector的大小
        return;
    }
    if (type == TZEROS) 
	{
		//std::cout << "init Blob:TZEROS" << endl;//debug:gordon
        data_ = vector<cube>(n, cube(h, w, c, fill::zeros));
        return;
    }
    if (type == TDEFAULT)
	{
		//std::cout << "init Blob:TDEFAULT" << endl;//debug:gordon
        data_ = vector<cube>(n, cube(h, w, c));
        return;
    }

	//2.填充Blob（实际上是一个vector<cube>变量）
    for (int i = 0; i < n; ++i)   //填充n个cube到Blob
	{
        cube tmp;  //立方体变量tmp，一张图片实际被表示为一个cube
        if (type == TRANDU) tmp = randu<cube>(h, w, c);//将元素设置为[0,1]区间内均匀分布的随机值
        if (type == TRANDN) tmp = randn<cube>(h, w, c);//使用μ=0和σ=1的高斯分布设置元素
        data_.push_back(tmp);  //叠加进去！
    }
    return;
}

