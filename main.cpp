#include <cmath>
#include <random> // C++11
#include <vector>
using namespace std;

// 시그모이드
double Sigmoid(double x){
	return 1 / (1 + exp(-x));
}

// 시그모이드 활성화 함수
double Sigmoid_Derivative(double x)
{
	double y = Sigmoid(x);
	return y * (1 - y);
}

// 뉴런 클래스
class Neuron
{
public:
	Neuron(size_t input_size)
	{
		Weights_.resize(input_size);
		count = input_size;
		Reset();
	}

public:
	// 결과값
	double Compute(const vector<double>& x) const
	{
		if (x.size() != Weights_.size())
			throw "x.size() != Weights_.size()";

		double wx = 0.0;
		for (size_t i = 0; i < Weights_.size(); ++i)
		{
			wx += Weights_[i] * x[i];
		}

		LastV_ = wx + Bias_;
		LastX_ = x;
		return Sigmoid(LastV_);
	}

	// NetWork 학습
	void Train(double a, const vector<pair<vector<double>, double>>& train_data)
	{
		size_t input_size = train_data[0].first.size();
		if (input_size != Weights_.size())
			throw "input_size != Weights_.size()";

		for (size_t i = 0; i < train_data.size(); ++i)
		{
			double o = Compute(train_data[i].first);
			double t = train_data[i].second;

			for (size_t j = 0; j < input_size; ++j)
			{
				Weights_[j] += a * Sigmoid_Derivative(LastV_)  * (t - o) * train_data[i].first[j];
			}
			Bias_ += a * Sigmoid_Derivative(LastV_)  * (t - o);
			LastD_ = Sigmoid_Derivative(LastV_) * (t - o);
		}
	}
	void Train(double a, double e, const vector<double>& train_data)
	{
		size_t input_size = train_data.size();
		if (input_size != Weights_.size())
			throw "input_size != Weights_.size()";

		for (size_t j = 0; j < input_size; ++j)
		{
			Weights_[j] += a * Sigmoid_Derivative(LastV_) * e * train_data[j];
		}
		Bias_ += a * Sigmoid_Derivative(LastV_) * e;

		LastD_ = Sigmoid_Derivative(LastV_) * e;
	}

	size_t InputSize() const
	{
		return Weights_.size();
	}
	double LastV() const
	{
		return LastV_;
	}
	double LastD() const
	{
		return LastD_;
	}
	vector<double>& Weights()
	{
		return Weights_;
	}
	double& Bias()
	{
		return Bias_;
	}
	const vector<double>& LastX() const
	{
		return LastX_;
	}



private:
	// 가중치 초기화
	void Reset()
	{
		Bias_ = -1;

		mt19937 random((random_device()()));
		uniform_real_distribution<double> dist(-2.4 / count, 2.4 / count);

		// -2.4/뉴런개수 ~ 2.4/ 뉴런개수 의 범위 랜덤값

		for (size_t i = 0; i < Weights_.size(); ++i)
		{
			Weights_[i] = dist(random);
		}
	}

private:
	int count;
	vector<double> Weights_;
	double Bias_;
	mutable double LastV_; // 가장 마지막으로 사용된 가중합을 저장
	double LastD_;
	mutable vector<double> LastX_;
};

class Network
{
public:

	// 레이어 초기화
	Network(const vector<size_t>& layers)
	{
		for (size_t i = 1; i < layers.size(); ++i)
		{
			vector<Neuron> layer;
			for (size_t j = 0; j < layers[i]; ++j)
			{
				layer.push_back(Neuron(layers[i - 1]));
			}
			Layers_.push_back(layer);
		}
	}

	// 출력 값을 추론,입력 값을 첫번째 레이어로 넘겨주고, 첫번째 레이어에 있는 인공 뉴런들의 출력 값을 벡터로 만들어서 다음 레이어로 넘겨주고.. 이걸 반복하도록 구현
	vector<double> Compute(const vector<double>& x){

		// 입력값의 사이즈 - 두번째 레이어의 첫번째 뉴런의 입력값 개수 의 값이 다르면 에러
		if (x.size() != Layers_[0][0].InputSize())
			throw "x.size() != Layers_[0][0].InputSize()";

		vector<double> result;
		vector<double> x_next_layer = x;

		for (size_t i = 0; i < Layers_.size(); ++i)
		{
			result.clear();
			for (size_t j = 0; j < Layers_[i].size(); ++j)
			{
				result.push_back(Layers_[i][j].Compute(x_next_layer));
			}
			x_next_layer = result;
		}

		return result;
	}
	void Train(double a, const vector<pair<vector<double>, vector<double>>>& train_data)
	{
		for (size_t i = 0; i < train_data.size(); ++i)
		{
			// 출력 레이어 학습
			vector<double> o = Compute(train_data[i].first);
			vector<double> e;

			if (o.size() != train_data[i].second.size())
				throw "o.size() != train_data[i].second.size()";

			for (size_t j = 0; j < o.size(); ++j)
			{
				e.push_back(train_data[i].second[j] - o[j]);
			}

			vector<double> d;
			for (size_t j = 0; j < Layers_[Layers_.size() - 1].size(); ++j)
			{
				Layers_[Layers_.size() - 1][j].Train(a, e[j], Layers_[Layers_.size() - 1][j].LastX());
				d.push_back(Layers_[Layers_.size() - 1][j].LastD());
			}

			if (Layers_.size() == 1)
				continue;

			// 은닉 레이어 학습
			for (size_t j = Layers_.size() - 2; j >= 0; --j)
			{
				vector<double> new_d;

				for (size_t k = 0; k < Layers_[j].size(); ++k)
				{
					vector<double> linked_w;
					for (size_t n = 0; n < Layers_[j + 1].size(); ++n)
					{
						linked_w.push_back(Layers_[j + 1][n].Weights()[k]);
					}

					if (linked_w.size() != d.size())
						throw "linked_w.size() != d.size()";

					double e_hidden = 0.0;
					for (size_t n = 0; n < linked_w.size(); ++n)
					{
						e_hidden += linked_w[n] * d[n];
					}

					Layers_[j][k].Train(a, e_hidden, Layers_[j][k].LastX());
					new_d.push_back(Layers_[j][k].LastD());
				}

				if (j == 0)
				{
					break;
				}

				d = new_d;
			}
		}
	}

private:
	vector<vector<Neuron>> Layers_;
};



#include <stdio.h>
#include <iostream>

int main(){

	// 입력 값이 2개, 첫번째 레이어의 인공 뉴런 개수는 2개,  두번째 레이어(출력 레이어)의 인공 뉴런 개수는 1개로 초기화 하는것입니다.
	Network net({ 2 , 2 , 1 });

	for (int i = 0; i < 1000000; ++i)
	{
		net.Train(0.2,
		{
			{ { 0, 0 }, { 0 } },
			{ { 1, 0 }, { 1 } },
			{ { 0, 1 }, { 1 } },
			{ { 1, 1 }, { 0 } },
		});

	}

	cout << "0 xor 0 = " << net.Compute({ 0, 0 })[0] << '\n';
	cout << "1 xor 0 = " << net.Compute({ 1, 0 })[0] << '\n';
	cout << "0 xor 1 = " << net.Compute({ 0, 1 })[0] << '\n';
	cout << "1 xor 1 = " << net.Compute({ 1, 1 })[0] << '\n';
	//cout << "Error = " << net.getError() << '\n';

	return 0;
}