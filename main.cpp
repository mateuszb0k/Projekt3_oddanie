#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <vector>
#include <cmath>
#include <matplot/matplot.h>
#include <set>
#include <sndfile.hh>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;
namespace m = matplot;
typedef std::complex<double> Complex;
std::vector<double> gen_sinus(int freq,int num_samples) {
    std::vector<double> y(num_samples);
	double step = 2 * 3.14159265359 * freq / num_samples;
	for (int i = 0; i < num_samples; ++i) {
		y[i] = sin(i * step);
	}
	return y;

}
std::vector<double> gen_cosinus(int freq, int num_samples) {
	std::vector<double> y(num_samples);
	double step = 2 * 3.14159265359 * freq / num_samples;
	for (int i = 0; i < num_samples; ++i) {
		y[i] = cos(i * step);
	}
	return y;

}
std::vector<double> gen_pila(int freq, int num_samples) {
	std::vector<double> y(num_samples);
	double step = 2 * 3.14159265359 * freq / num_samples;
	for (int i = 0; i < num_samples; ++i) {
		y[i] = std::fmod(i * step, 2 * 3.14159265359);
	}
	return y;

}
std::vector<double> gen_prostokatny(int freq, int num_samples) {
	std::vector<double> y(num_samples);
	double step = 2 * 3.14159265359 * freq / num_samples;
	for (int i = 0; i < num_samples; ++i) {
		y[i] = sin(i * step) > 0 ? 1 : -1;
	}
	return y;

}
void sinus(int freq, int num_samples = 1000) {
    std::vector<double> x(num_samples);
    double step = 2 * 3.14159265359 * freq / num_samples;
    for (int i = 0; i < num_samples; ++i) {
        x[i] = i * step;
    }
    m::plot(x, gen_sinus(freq,num_samples));
    m::show();
}

void cosinus(int freq, int num_samples = 1000) {
    std::vector<double> x(num_samples);
	double step = 2 * 3.14159265359 * freq / num_samples;
	for (int i = 0; i < num_samples; ++i) {
		x[i] = i * step;
	}
	m::plot(x, gen_cosinus(freq, num_samples));
	m::show();
}

void pila(int freq, int num_samples = 1000) {
	std::vector<double> x(num_samples);
	double step = 2 * 3.14159265359 * freq / num_samples;
	for (int i = 0; i < num_samples; ++i) {
		x[i] = i * step;
	}
	m::plot(x, gen_pila(freq, num_samples));
	m::show();
}

void prostokatny(int freq, int num_samples = 1000) {
    std::vector<double> x(num_samples);
	double step = 2 * 3.14159265359 * freq / num_samples;
	for (int i = 0; i < num_samples; ++i) {
		x[i] = i * step;
	}
	m::plot(x, gen_prostokatny(freq, num_samples));
	m::show();
}
void visualize_wav(const std::string& filename, int downsample_factor = 10) {
    SndfileHandle file { filename };
    if (file.error()) {
        std::cerr << "Error: blad pliku\n";
        return;
    }
    if (file.channels() > 2) {
        std::cerr << "Error: tylko pliki mono i stereo\n";
        return;
    }
    size_t n_frames = file.frames();
    std::vector<double> data(n_frames * file.channels());
    file.read(data.data(), n_frames);
    std::vector<double> time(n_frames);
    std::vector<double> downsampled_data;
    std::vector<double> downsampled_time;

    for (size_t i = 0; i < n_frames; i += downsample_factor) {
        downsampled_time.push_back(static_cast<double>(i) / file.samplerate());
        downsampled_data.push_back(data[i]);
    }
    matplot::plot(downsampled_time, downsampled_data);
    matplot::show();
}
std::vector<Complex> DFT(const std::vector<double>& input) {
    int N = input.size();
    std::vector<Complex> output(N);
    for (int k = 0; k < N; ++k) {
        for (int n = 0; n < N; ++n) {
            double theta = 2.0 * M_PI * k * n / N;
            output[k] += input[n] * Complex(cos(theta), -sin(theta));
        }
    }
    return output;
}
std::vector<Complex> IDFT(const std::vector<Complex>& input) {
    int N = input.size();
    std::vector<Complex> output(N);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < N; ++k) {
            double theta = 2.0 * M_PI * k * n / N;
            output[n] += input[k] * Complex(cos(theta), sin(theta));
        }
        output[n] /= N;
    }
    return output;
}
void visualize_DFT(const std::vector<double>& input) {
    std::vector<Complex> output = DFT(input);
    std::vector<double> frequencies(output.size());
    std::vector<double> magnitudes(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        frequencies[i] = i;
        magnitudes[i] = std::abs(output[i]);
    }
    matplot::plot(frequencies, magnitudes);
    matplot::xlabel("Częstotliwosc");
    matplot::ylabel("Amplituda");
    matplot::show();
}
std::vector<double> filter1D(const std::vector<double>& input,  std::vector<double>& kernel) {
    if (kernel.size() % 2 == 0) {
        kernel.insert(kernel.begin(), 0);
    }
    int kernelSize = kernel.size();
    int dataSize = input.size();
    std::vector<double> output(dataSize);
    int kernelCenter = kernelSize / 2;  

    for (int i = 0; i < dataSize; ++i) {
        double sum = 0;
        for (int j = 0; j < kernelSize; ++j) {
            int ii = i - kernelCenter + j;
            if (ii < 0 || ii >= dataSize) continue;
            sum += input[ii] * kernel[j];
        }
        output[i] = sum;
    }

    return output;
}
std::vector<std::vector<std::vector<double>>> filter2D(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<double>>& kernel) {
    int kernelSize = kernel.size();
    int height = input.size();
    int width = input[0].size();
    int channels = input[0][0].size();
    if(channels == NULL || channels ==0) channels = 1;
    std::vector<std::vector<std::vector<double>>> output(height, std::vector<std::vector<double>>(width, std::vector<double>(channels)));

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                double sum = 0;
                for (int ki = 0; ki < kernelSize; ++ki) {
                    for (int kj = 0; kj < kernelSize; ++kj) {
                        int ii = i - ki;
                        int jj = j - kj;
                        if (ii < 0 || jj < 0 || ii >= height || jj >= width) continue;
                        if (channels != 1) {
                            sum += input[ii][jj][c] * kernel[ki][kj];
                        }
						else {
							sum += input[ii][jj][0] * kernel[ki][kj];
						}
                    }
                }
                output[i][j][c] = sum;
            }
        }
    }

    return output;
}
std::vector<std::vector<double>> edgeDetectionKernel() {
	return {
		{ -1, -1, -1 },
		{ -1,  8, -1 },
		{ -1, -1, -1 }
	};
}
std::vector<std::vector<double>> boxblurKernel() {
	return {
		{ 1.0 / 9, 1.0 / 9, 1.0 / 9 },
		{ 1.0 / 9, 1.0 / 9, 1.0 / 9 },
		{ 1.0 / 9, 1.0 / 9, 1.0 / 9 }
	};
}
std::vector<std::vector<double>> sharpenKernel() {
	return {
		{  0, -1,  0 },
		{ -1,  5, -1 },
		{  0, -1,  0 }
	};
}
std::vector<std::vector<double>> gaussianBlurKernel() {
	return {
		{ 1.0 / 16, 2.0 / 16, 1.0 / 16 },
		{ 2.0 / 16, 4.0 / 16, 2.0 / 16 },
		{ 1.0 / 16, 2.0 / 16, 1.0 / 16 }
	};
}

PYBIND11_MODULE(cmake_example, m) {
	m.def("gen_sinus", &gen_sinus);
	m.def("gen_cosinus", &gen_cosinus);
	m.def("gen_pila", &gen_pila);
	m.def("gen_prostokatny", &gen_prostokatny);
    m.def("sinus", &sinus);
    m.def("cosinus", &cosinus);
    m.def("pila", &pila);
    m.def("prostokatny", &prostokatny);
    m.def("visualize_wav", &visualize_wav);
    m.def("visualize_DFT", &visualize_DFT);
	m.def("filter1D", &filter1D);
	m.def("filter2D", &filter2D);
	m.def("edgeDetectionKernel", &edgeDetectionKernel);
	m.def("boxblurKernel", &boxblurKernel);
	m.def("sharpenKernel", &sharpenKernel);
	m.def("gaussianBlurKernel", &gaussianBlurKernel);
}
