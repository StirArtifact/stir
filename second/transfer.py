import subprocess
import os

def recover_data(rpath1, rpath2, wpath):
	# cmd = "./PCFG/pcfg.o"
	cmd = "./second/pcfg/pcfg.exe" if os.name == 'nt' else "./second/pcfg/pcfg.o"
	with open(wpath,'w',encoding='utf-8') as file_w:
		linet = open(rpath1,'r',encoding='utf-8')
		linep = open(rpath2,'r',encoding='utf-8')
		for line1, line2 in zip(linet, linep):
		# for line in open(rpath,'r',encoding='utf-8'):
			splits1 = line1.strip().split('\t')
			splits2 = line2.strip().split('\t')
			if(len(splits1) == 1):
				out_str = splits2[0]+'\n'
			else:
				splits2 = splits2[:10]+['<eos>']*10
				data = '\t'.join(splits2)+'\n'
				output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, input=data.encode('utf-8')).decode('utf-8')
				out_str = output.strip() + '\n'
				file_w.write(out_str)


if __name__ == '__main__':
	rpt = "../test.lf.data_"
	rpp = "../test.infer.ner"
	wp = "../test_infer_ner.txt"

	recover_data(rpt, rpp, wp)
