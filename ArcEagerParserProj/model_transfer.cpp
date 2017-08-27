#include "model_transfer.h"
#include "common.h"
#include "TreeReader.h"

void parameters_store_log(parameters_store& ps)
{
	printf("---------------------\nbegin checking\n");
	for (auto iter = ps.name2Parameters_state.begin(); iter != ps.name2Parameters_state.end(); iter++) {
		if (iter->second) {
			printf("Warning: Parameters (%s) did not init!\n", iter->first.c_str());
		}
	}	for (auto iter = ps.name2lookupParameters_state.begin(); iter != ps.name2lookupParameters_state.end(); iter++) {
		if (iter->second) {
			printf("Warning: LookupParameters (%s) did not init!\n", iter->first.c_str());
		}
	}
	printf("Finish checking\n------------------------\n");
}

void load(const char* path, map<string,string>& locvar2orgvar) {
	std::ifstream tbFile(path);
	std::string lineS;

	char* buffer = (char*)malloc(2048);
	int buffer_size = 2048;

	while (getline(tbFile, lineS)) {
		if (lineS.size() >= buffer_size) {
			free(buffer);
			buffer_size = 2 * lineS.size();
			buffer = (char*)malloc(buffer_size);
		}
		strcpy(buffer, lineS.c_str());
		//--------------------------------------
		char* ln = strtrim(buffer);
		if (ln[0] == 0)
			continue;

		//element
		vector<char*> eles;
		split2(ln, '\t', eles);
		if (eles.size() < 2) {
			printf("info: var (%s) missing corresponding var\n",eles[0]);
			continue;
		}
		locvar2orgvar[eles[0]] = eles[1];
	}
	
	free(buffer);
}

#define check_map(iter, mapobj,fmt,msg) if((iter)==(mapobj).end()) {printf(fmt, msg);exit(-1);}
#define check_map_warning(iter, mapobj,fmt,msg) if((iter)==(mapobj).end()) {printf(fmt, msg);continue;}

void transfer_args(parameters_store& ps, Model& mdl, const char* inifile)
{
	map<string, string> locvar2orgvar;
	load(inifile, locvar2orgvar);
	for (auto iter = ps.name2lookupParameters_state.begin();iter != ps.name2lookupParameters_state.end();iter++) {
		if (!iter->second)
			continue;
		auto titer = locvar2orgvar.find(iter->first);
		check_map(titer, locvar2orgvar, "ERROR: var(%s) missing in inifile!\n", iter->first.c_str());
		auto iter1 = ps.name2lookupParameters.find(iter->first);
		auto iter2 = ps.name2lookupParameters_Unused.find(titer->second);
		check_map(iter1, ps.name2lookupParameters, "ERROR: var(%s) missing in name2lookupParameters!\n", iter->first.c_str());
		check_map(iter2, ps.name2lookupParameters_Unused, "ERROR: var(%s) missing in name2lookupParameters_Unused!\n", titer->second.c_str());
		(*iter1->second).copy(*iter2->second);

		iter->second = false;
	}

	for (auto iter = ps.name2Parameters_state.begin();iter != ps.name2Parameters_state.end();iter++) {
		if (!iter->second)
			continue;
		auto titer = locvar2orgvar.find(iter->first);
		check_map_warning(titer, locvar2orgvar, "\twarning: var(%s) missing in inifile!\n", iter->first.c_str());
		auto iter1 = ps.name2Parameters.find(iter->first);
		auto iter2 = ps.name2Parameters_Unused.find(titer->second);
		check_map(iter1, ps.name2Parameters, "ERROR: var(%s) missing in name2Parameters!\n", iter->first.c_str());
		if (iter2 == ps.name2Parameters_Unused.end()) {
			iter2 = ps.name2Parameters.find(titer->second);
			check_map(iter2, ps.name2Parameters, "ERROR: var(%s) missing in name2Parameters(for Unused)!\n", titer->second.c_str());
			printf("warining: var(%s) missing in name2Parameters_Unused\n", titer->second.c_str());
		}
		check_map(iter2, ps.name2Parameters_Unused, "ERROR: var(%s) missing in name2Parameters_Unused!\n", titer->second.c_str());
		(*iter1->second).copy(*iter2->second);

		iter->second = false;
	}
	parameters_store_log(ps);
}

////////////////////////////////////////////////////////////////////////////////////////
void load_string2idx(const char* path, map<string, unsigned>& locvar2orgvar) {
	std::ifstream tbFile(path);
	std::string lineS;

	char* buffer = (char*)malloc(2048);
	int buffer_size = 2048;

	while (getline(tbFile, lineS)) {
		if (lineS.size() >= buffer_size) {
			free(buffer);
			buffer_size = 2 * lineS.size();
			buffer = (char*)malloc(buffer_size);
		}
		strcpy(buffer, lineS.c_str());
		//--------------------------------------
		char* ln = strtrim(buffer);
		if (ln[0] == 0)
			continue;

		//element
		vector<char*> eles;
		split2(ln, '\t', eles);
		if (eles.size() < 2) {
			printf("info: var (%s) missing corresponding var\n", eles[0]);
			continue;
		}
		locvar2orgvar[eles[0]] = atoi(eles[1]);
	}

	free(buffer);
}

void save_string2idx(const char* path, map<string, unsigned>& locvar2orgvar)
{
	FILE* fout = fopen(path, "wb");
	for (auto iter : locvar2orgvar) {
		fprintf(fout, "%s\t%d\n", iter.first.c_str(), iter.second);
	}
	fclose(fout);
}

int fill_wordEM(LookupParameters& useEM, LookupParameters& orgEM, map<string, unsigned>& org_word2idx, map<string, unsigned>& cur_word2idx)
{
	int num_init = 0;
	for (auto org_i : org_word2idx) {
		auto cur_i = cur_word2idx.find(org_i.first);
		if (cur_i == cur_word2idx.end())
			continue;
		int orgi = org_i.second;
		int curi = cur_i->second;
		if (curi >= useEM.values.size() || orgi >= orgEM.values.size())
			continue;

		TensorTools::CopyElements(useEM.values[curi], orgEM.values[orgi]);
		num_init++;
	}
	return num_init;
}

void convert_args(parameters_store & ps, Model & mdl, const char * inifile, map<string, unsigned>& org_word2idx, map<string, unsigned>& cur_word2idx)
{
	float tt_num = 0,tt_numorg=0, init_num = 0;
	map<string, string> locvar2orgvar;
	load(inifile, locvar2orgvar);
	for (auto iter = ps.name2lookupParameters_state.begin(); iter != ps.name2lookupParameters_state.end(); iter++) {
		if (!iter->second)
			continue;
		auto titer = locvar2orgvar.find(iter->first);
		if (titer == locvar2orgvar.end())
			continue;
		auto iter1 = ps.name2lookupParameters.find(iter->first);
		auto iter2 = ps.name2lookupParameters_Unused.find(titer->second);
		if (iter1 == ps.name2lookupParameters.end() || iter2 == ps.name2lookupParameters_Unused.end()) {
			printf("ERROR: var(%s) missing in name2lookupParameters! OR\n", iter->first.c_str());
			printf("ERROR: var(%s) missing in name2lookupParameters_Unused!\n", titer->second.c_str());
			exit(-1);
		}

		init_num+=fill_wordEM(*iter1->second, *iter2->second, org_word2idx, cur_word2idx);
		tt_num += iter1->second->values.size();
		tt_numorg+= iter2->second->values.size();

		iter->second = false;
	}
	printf("convert_args: using %d, total1 %d, total2 %d, done1 %.3f, done2 %.3f\n", (int)init_num, (int)tt_num,(int)tt_numorg,
		init_num / tt_num, init_num / tt_numorg);
}
