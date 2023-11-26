#include <iostream>
#include <map>
#include <fstream>
#include "csvstream.hpp"
#include <cmath>
#include <utility>
#include <set>

using namespace std;

class Classifier
{
public:
	Classifier(const string &training_file_in, const string &test_file_in, const bool &debug_flag_in)
		: training_file(training_file_in), test_file(test_file_in), debug_flag(debug_flag_in) {}

	void training()
	{
		string filename = training_file;
		try
		{
			csvstream csvin(filename);
			map<string, string> row;
			string all_vocab;
			if (debug_flag)
			{
				cout << "training data:" << endl;
			}
			while (csvin >> row)
			{
				string label = row["tag"];
				string content = row["content"];
				num_of_posts++;

				all_vocab = all_vocab + content + " ";
				num_of_post_per_w(content);
				num_post_post_with_C_with_W(label, content);

				labels[label]++;
				if (debug_flag)
				{
					cout << "  label = " << label;
					cout << ", content = "
						 << content << endl;
				}
			}
			uniq_words = unique_words(all_vocab);
			vocab_size = uniq_words.size();
			calc_log_prior();
			// calc_log_likely(uniq_words);
			if (debug_flag)
			{
				test_debug();
			}
		}
		catch (const csvstream_exception &e)
		{
			cout << "Error opening file: " << filename << endl;
			return;
		}
	}

	void testing()
	{
		string filename = test_file;
		int num_of_test_post = 0;
		int num_of_correct_pred = 0;
		try
		{
			csvstream csvin(filename);
			map<string, string> row;
			if (!debug_flag)
			{
				cout << "trained on " << num_of_posts << " examples" << endl;
				cout << "\n";
			}
			cout << "test data:" << endl;
			while (csvin >> row)
			{
				string label = row["tag"];
				string content = row["content"];

				set<string> uniq_words_in_content = unique_words(content);
				string prediced_label = label_prediction(uniq_words_in_content);
				if (prediced_label == label)
				{
					num_of_correct_pred++;
				}

				cout << "  correct = " << label << ","
					 << " predicted = " << prediced_label
					 << ","
					 << " log-probability score = " << calc_log_prob(uniq_words_in_content, prediced_label) << endl;
				cout << "  content = "
					 << content << endl;
				cout << "\n";

				num_of_test_post++;
			}

			cout << "performance: " << num_of_correct_pred << " / " << num_of_test_post << " posts predicted correctly" << endl;

			// for (const auto &kv : words)
			// {
			// 	cout << kv.first;
			// 	cout << kv.second << endl;
			// }
		}
		catch (const csvstream_exception &e)
		{
			cout << "Error opening file: " << filename << endl;
			return;
		}
	}

	void test_debug()
	{
		cout << "trained on " << num_of_posts << " examples" << endl;
		cout << "vocabulary size = " << vocab_size << endl;
		cout << "\n";

		cout << "classes:" << endl;
		for (const auto &kv1 : labels)
		{
			cout << "  " << kv1.first << ", " << kv1.second << " examples, "
				 << "log-prior = " << log_prior[kv1.first] << endl;
		}

		cout << "classifier parameters:" << endl;
		for (const auto &kv2 : word_per_label)
		{
			for (const auto &kv3 : kv2.second)
			{
				cout << "  " << kv2.first << ":";
				cout << kv3.first << ", count = " << kv3.second << ", ";
				cout << "log-likelihood = " << calc_log_likely(kv3.first, kv2.first) << endl;
			}
		}
		cout << "\n";
	}

private:
	string training_file;
	string test_file;
	bool debug_flag;
	int num_of_posts = 0;
	int vocab_size = 0;
	map<string, int> labels;
	map<string, int> words;
	set<string> uniq_words;
	map<string, map<string, int>> word_per_label;
	map<string, double> log_prior;
	// label, word, log_likely
	// map<string, map<string, double>> log_likely;
	// label, log_prob

	// EFFECTS: Return a set of unique whitespace delimited words.x
	set<string>
	unique_words(const string &str)
	{
		istringstream source(str);
		set<string> words;
		string word;
		while (source >> word)
		{
			words.insert(word);
		}
		return words;
	}

	void num_of_post_per_w(string &content)
	{
		set<string> words_in_content = unique_words(content);
		for (set<string>::iterator i = words_in_content.begin(); i != words_in_content.end(); ++i)
		{
			words[*i]++;
		}
	}

	void num_post_post_with_C_with_W(string &label, string &content)
	{
		set<string> words_in_content = unique_words(content);
		for (set<string>::iterator i = words_in_content.begin(); i != words_in_content.end(); ++i)
		{
			word_per_label[label][*i]++;
		}
	}

	void calc_log_prior()
	{
		for (const auto &kv : labels)
		{
			double log_p = log(double(kv.second) / double(num_of_posts));
			log_prior.insert({kv.first, log_p});
		}
	}

	double calc_log_likely(string content, string label)
	{
		double log_l;
		if (word_per_label[label][content] == 0)
		{
			if (words[content] == 0)
			{
				log_l = log(1 / double(num_of_posts));
				// cout << "C";
			}
			else
			{
				log_l = log(double(words[content]) / double(num_of_posts));
				// cout << "B";
			}
		}
		else
		{
			log_l = log(double(word_per_label[label][content]) / double(labels[label]));
			// cout << "A";
		}

		return log_l;
	}

	double calc_log_prob(set<string> &content, string label)
	{
		double log_p = log_prior[label];
		for (set<string>::iterator i = content.begin(); i != content.end(); ++i)
		{
			log_p += calc_log_likely(*i, label);
		}
		// cout << log_p;
		return log_p;
	}

	string label_prediction(set<string> &content)
	{

		map<string, double> log_prob;
		for (const auto &kv1 : labels)
		{
			log_prob.insert({kv1.first, calc_log_prob(content, kv1.first)});
			// cout << calc_log_prob(kv1.first, content) << endl;
		}
		map<string, double>::iterator i = log_prob.begin();
		double max_prob = i->second;
		string max_label = i->first;
		// cout << max_label << ": " << max_prob << endl;
		for (const auto &kv2 : log_prob)
		{
			// cout << kv2.second;
			if (kv2.second > max_prob)
			{
				max_prob = kv2.second;
				// cout << labels.size() << endl;
				max_label = kv2.first;
			}
		}
		// cout << endl;
		// cout << max_label << ": " << max_prob << endl;
		return max_label;
	}
};

int main(int argc, char **argv)
{
	cout.precision(3);

	if (argc != 3 && argc != 4)
	{
		cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
		return 1;
	}
	string debug_flag;

	if (argc == 4)
	{
		debug_flag = argv[3];
		if (debug_flag != "--debug")
		{
			cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
			return 1;
		}
	}

	string training_file = argv[1];
	string test_file = argv[2];

	if (argc == 4)
	{
		Classifier classifier(training_file, test_file, 1);
		classifier.training();
		classifier.testing();
	}
	else
	{
		Classifier classifier(training_file, test_file, 0);
		classifier.training();
		classifier.testing();
	}

	return 0;
}
