#include "cnn/model.h"
#include "cnn/tensor.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/cnn.h"

#include <unordered_set>
#include <iostream>

#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#define CNN_ALIGN 256
#if HAVE_CUDA
#include "cnn/gpu-ops.h"
#include "cnn/cuda.h"
#endif

using namespace std;

namespace cnn {

ParametersBase::~ParametersBase() {}

Parameters::Parameters(const Dim& d, float scale) : dim(d) {
  values.d = g.d = d;
  values.v = (float*)cnn_mm_malloc(d.size() * sizeof(float), CNN_ALIGN);
  if (scale) TensorTools::Randomize(values, scale); else TensorTools::Randomize(values);
  g.v = (float*)cnn_mm_malloc(d.size() * sizeof(float), CNN_ALIGN);
  TensorTools::Zero(g);
}

size_t Parameters::size() const { return dim.size(); }

void Parameters::scale_parameters(float a) {
  (*g) *= a;
}

void Parameters::squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  gpu::l2_norm_reducer(values.d.size(), values.v, sqnorm, true, false);
#else
  *sqnorm = (*values).squaredNorm();
#endif
}

void Parameters::g_squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  gpu::l2_norm_reducer(g.d.size(), g.v, sqnorm, true, false);
#else
  *sqnorm = (*g).squaredNorm();
#endif
}

void Parameters::copy(const Parameters & param) {
  assert(dim == param.dim);
  TensorTools::CopyElements(values, param.values);
}

void Parameters::accumulate_grad(const Tensor& d) {
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, g.d.size(), kSCALAR_ONE, d.v, 1, g.v, 1));
#else
  *g += *d;
#endif
}

void Parameters::clear() {
  TensorTools::Zero(g);
}

LookupParameters::LookupParameters(unsigned n, const Dim& d) : dim(d), values(n), grads(n) {
  for (unsigned i = 0; i < n; ++i) {
    auto& v = values[i];
    v.d = d;
    v.v = (float*)cnn_mm_malloc(d.size() * sizeof(float), CNN_ALIGN);
    TensorTools::Randomize(v);

    auto& g = grads[i];
    g.d = d;
    g.v = (float*)cnn_mm_malloc(d.size() * sizeof(float), CNN_ALIGN);
    TensorTools::Zero(g);
  }
}

void LookupParameters::scale_parameters(float a) {
  for (auto& p : values)
    (*p) *= a;
}

void LookupParameters::Initialize(unsigned index, const vector<float>& val) {
  assert(int(val.size()) == int(dim.size()));
#if HAVE_CUDA
  cerr << "implement LookupParameters::Initialize\n";
  throw cuda_not_implemented("LookupParameters::Initialize");
#else
  memcpy(values[index].v, &val[0], val.size() * sizeof(float));
#endif
}

size_t LookupParameters::size() const {
  return values.size() * dim.size();
}

void LookupParameters::g_squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  bool acc = false;
  for (auto i : non_zero_grads) {
    gpu::l2_norm_reducer(grads[i].d.size(), grads[i].v, sqnorm, true, acc);
    acc = true;
  }
#else
  real a = 0;
  for (auto i : non_zero_grads)
    a += (*grads[i]).squaredNorm();
  *sqnorm = a;
#endif
}

void LookupParameters::squared_l2norm(float* sqnorm) const {
#if HAVE_CUDA
  bool acc = false;
  for (unsigned i = 0; i < values.size(); ++i) {
    gpu::l2_norm_reducer(values[i].d.size(), values[i].v, sqnorm, true, acc);
    acc = true;
  }
#else
  float a = 0;
  for (unsigned i = 0; i < values.size(); ++i)
    a += (*values[i]).squaredNorm();
  *sqnorm = a;
#endif
}

void LookupParameters::copy(const LookupParameters & param) {
  assert(dim == param.dim);
  for(size_t i = 0; i < param.values.size(); ++i)
    TensorTools::CopyElements(values[i], param.values[i]);
}

void LookupParameters::accumulate_grad(unsigned index, const Tensor& d) {
  non_zero_grads.insert(index);
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, d.d.size(), kSCALAR_ONE, d.v, 1, grads[index].v, 1));
#else
  *grads[index] += *d;
#endif
}

void LookupParameters::clear() {
  for (auto i : non_zero_grads)
    TensorTools::Zero(grads[i]);
  non_zero_grads.clear();
}

Model::~Model() {
  for (auto p : all_params) delete p;
}

void Model::project_weights(float radius) {
  static float* project_scratch = 0;
  if (!project_scratch)
    project_scratch = (float*)cnn_mm_malloc(all_params.size() * sizeof(float), 256);
  int pi = 0;
  for (auto p : all_params) {
    p->squared_l2norm(&project_scratch[pi]);
    ++pi;
  }
  double gg = 0;
  for (int i = 0; i < pi; ++i)
    gg += project_scratch[i];
  cerr << "NORM: " << sqrt(gg) << endl;
}

float Model::gradient_l2_norm() const {
  if (!gradient_norm_scratch)
    gradient_norm_scratch = (float*)cnn_mm_malloc(all_params.size() * sizeof(float), 256);
  int pi = 0;
  for (auto p : all_params) {
    p->g_squared_l2norm(&gradient_norm_scratch[pi]);
    ++pi;
  }
#if HAVE_CUDA
  float res = 0;
  gpu::l2_norm_reducer(all_params.size(), gradient_norm_scratch, gradient_norm_scratch, false, false);
  cudaMemcpy(&res, gradient_norm_scratch, sizeof(float),  cudaMemcpyDeviceToHost);
  return sqrt(res);
#else
  double gg = 0;
  for (int i = 0; i < pi; ++i)
    gg += gradient_norm_scratch[i];
  return sqrt(gg);
#endif
}

Parameters* Model::add_parameters(const Dim& d, std::string name, float scale) {
  Parameters* p = new Parameters(d, scale);
  all_params.push_back(p);
  params.push_back(p);
  p->set_name(name_scope+"/"+name);
  return p;
}

LookupParameters* Model::add_lookup_parameters(unsigned n, const Dim& d, std::string name) {
  LookupParameters* p = new LookupParameters(n,d);
  all_params.push_back(p);
  lookup_params.push_back(p);
  p->set_name(name_scope + "/" + name);
  return p;
}

void Model::reset_gradient() {
  for (auto p : params) { p->clear(); }
  for (auto p : lookup_params) { p->clear(); }
}

void save_cnn_model(std::string filename, Model* model) {
    std::ofstream out(filename);
    boost::archive::text_oarchive oa(out);
    oa << (*model);
};

void load_cnn_model(std::string filename, Model* model) {
    std::ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> (*model);
};
//-----------------------------------------------


void save_cnn_model_byname(std::string filename, Model* model) 
{
	std::ofstream out_txt(filename + ".log");

	std::ofstream out_para(filename + ".params");
	boost::archive::text_oarchive oa_para(out_para);
	std::vector<Parameters*>& params = model->params;

	int np = params.size();
	map<string, Parameters*> name2Parameters;
	for (unsigned i = 0; i < params.size(); ++i)
	{
		string name = params[i]->para_name;
		if (name2Parameters.find(name) != name2Parameters.end()) {
			printf("Error: params (%s) conflict!\n", name.c_str());
			exit(-1);
		}
		name2Parameters[name] = params[i];
	}
	oa_para & np;
	for (auto iter = name2Parameters.begin();iter != name2Parameters.end();iter++) {
		oa_para & iter->first;
		oa_para & *iter->second;
		out_txt << "parameters:\t" << iter->first<<endl;
	}
	//------------------------------------------
	std::ofstream out_lookuppara(filename + ".lookupparams");
	boost::archive::text_oarchive oa_lookuppara(out_lookuppara);

	std::vector<LookupParameters*>& lookup_params = model->lookup_params;
	int nlp = lookup_params.size();
	map<string, LookupParameters*> name2lookupParameters;
	for (unsigned i = 0; i < lookup_params.size(); ++i)
	{
		string name = lookup_params[i]->para_name;
		if (name2lookupParameters.find(name) != name2lookupParameters.end()) {
			printf("Error: lookup_params (%s) conflict!\n", name.c_str());
			exit(-1);
		}
		name2lookupParameters[name] = lookup_params[i];
	}

	oa_lookuppara & nlp;
	for (auto iter = name2lookupParameters.begin();iter != name2lookupParameters.end();iter++) {
		oa_lookuppara & iter->first;
		oa_lookuppara & *iter->second;
		out_txt <<"lookup parameters:\t"<< iter->first << endl;
	}
};
//-----------------------------------------------

void load_LookupParameters(boost::archive::text_iarchive& ar, LookupParameters& par)
{
	ar & par.dim;
	int nv;
	ar & nv;
	par.values.resize(nv);
	for (unsigned i = 0; i < par.values.size(); ++i)
		ar & par.values[i];
}

void load_cnn_model_byname(std::string filename, parameters_store& ps, Model* model) {
	printf("Begin Loading(%s)...\n", filename.c_str());
	std::vector<Parameters*>& params = model->params;
	int np = params.size();
	map<string, Parameters*>& name2Parameters= ps.name2Parameters;
	map<string, bool>& name2Parameters_state = ps.name2Parameters_state;
	for (unsigned i = 0; i < params.size(); ++i)
	{
		string name = params[i]->para_name;
		if (name2Parameters.find(name) != name2Parameters.end()) {
			printf("Error: params (%s) conflict!\n", name.c_str());
			exit(-1);
		}
		name2Parameters[name] = params[i];
		name2Parameters_state[name] = true;
	}
	
	std::vector<LookupParameters*>& lookup_params = model->lookup_params;
	int nlp = lookup_params.size();
	map<string, LookupParameters*>& name2lookupParameters = ps.name2lookupParameters;
	map<string, bool>& name2lookupParameters_state = ps.name2lookupParameters_state;
	for (unsigned i = 0; i < lookup_params.size(); ++i)
	{
		string name = lookup_params[i]->para_name;
		if (name2lookupParameters.find(name) != name2lookupParameters.end()) {
			printf("Error: lookup_params (%s) conflict!\n", name.c_str());
			exit(-1);
		}
		name2lookupParameters[name] = lookup_params[i];
		name2lookupParameters_state[name] = true;
	}
	//------------------------------------------------------------


	std::ifstream in1(filename + ".params");
	boost::archive::text_iarchive ia1(in1);
	int np_ld;
	ia1 & np_ld;
	map<string, Parameters*>& name2Parameters_Unused = ps.name2Parameters_Unused;
	for (int i = 0;i < np_ld;i++) {
		string name;
		ia1 & name;
		auto iname = name2Parameters_state.find(name);
		if (iname != name2Parameters_state.end() && iname->second == false) {
			printf("Error in Loading: Parameters (%s) conflict!\n", name.c_str());
			exit(-1);
		}

		auto iter = name2Parameters.find(name);
		if (iter != name2Parameters.end()) {
			ia1 & *iter->second;
		} else {
			Parameters* un_used = new Parameters();
			ia1 & *un_used;
			un_used->set_name(name);
			name2Parameters_Unused[name] = un_used;
			printf("Warning in Loading: Parameters (%s) missing!\n", name.c_str());
		}
		name2Parameters_state[name] = false;
	}
	//-------------------------------------------
	std::ifstream in2(filename + ".lookupparams");
	boost::archive::text_iarchive ia2(in2);
	int nlp_ld; ia2 & nlp_ld;
	map<string, LookupParameters*>& name2lookupParameters_Unused = ps.name2lookupParameters_Unused;

	for (int i = 0;i < nlp_ld;i++) {
		string name;
		ia2 & name;
		auto iname = name2lookupParameters_state.find(name);
		if (iname != name2lookupParameters_state.end() && iname->second == false) {
			printf("Error in Loading: Parameters (%s) conflict!\n", name.c_str());
			exit(-1);
		}

		auto iter = name2lookupParameters.find(name);
		if (iter != name2lookupParameters.end()) {
			ia2 & *iter->second;
		}
		else {
			LookupParameters* un_used = new LookupParameters();
			load_LookupParameters(ia2, *un_used);
			un_used->set_name(name);
			name2lookupParameters_Unused[name] = un_used;
			printf("Warning in Loading: Parameters (%s) missing!\n", name.c_str());
		}
		name2lookupParameters_state[name] = false;
	}
	//-------------------------------------------
	for (auto iter = name2Parameters_state.begin();iter != name2Parameters_state.end();iter++) {
		if (iter->second) {
			printf("Warning: Parameters (%s) did not init!\n", iter->first.c_str());
		}
	}	for (auto iter = name2lookupParameters_state.begin();iter != name2lookupParameters_state.end();iter++) {
		if (iter->second) {
			printf("Warning: LookupParameters (%s) did not init!\n", iter->first.c_str());
		}
	}
	printf("Finish Loading (%s)\n", filename.c_str());
};

} // namespace cnn
