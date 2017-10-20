
time /home/zgy_ucla_cs/tools/metaphlan2/biobakery-metaphlan2-f1dcf3958459/metaphlan2.py --bowtie2_exe /home/zgy_ucla_cs/tools/bowtie2-2.2.9/bowtie2 --bowtie2out /home/zgy_ucla_cs/MetaSUB/bowtie/SRR1748536_1.bz2 /home/zgy_ucla_cs/MetaSUB/SRR1748536.fastq_1 --input_type fastq  > /home/zgy_ucla_cs/MetaSUB/profile/profiled_SRR1748536_1.txt



time /home/zgy_ucla_cs/tools/metaphlan2/biobakery-metaphlan2-f1dcf3958459/metaphlan2.py --bowtie2db /home/zgy_ucla_cs/tools/metaphlan2/biobakery-metaphlan2-f1dcf3958459/db_v20/mpa_v20_m200 --bowtie2_exe /home/zgy_ucla_cs/tools/bowtie2-2.2.9/bowtie2 --bowtie2out /home/zgy_ucla_cs/MetaSUB/bowtie/SRR1748535.bz2.txt /home/zgy_ucla_cs/MetaSUB/SRR1748535.fastq --input_type fastq  > /home/zgy_ucla_cs/MetaSUB/profile/profiled_SRR1748535.txt


profiled_SRR1748535
real    2m7.408s
user    2m0.650s
sys     0m4.460s


-----------
metaphlan2.py --bowtie2db /home/zgy_ucla_cs/tools/metaphlan2/biobakery-metaphlan2-f1dcf3958459/db_v20/mpa_v20_m200 --bowtie2_exe /home/zgy_ucla_cs/tools/bowtie2-2.2.9/bowtie2 --bowtie2out /home/zgy_ucla_cs/MetaSUB/bowtie/SRR1748535.bz2.txt /home/zgy_ucla_cs/MetaSUB/SRR1748535.fastq --input_type fastq --mpa_pkl /home/zgy_ucla_cs/tools/metaphlan2/biobakery-metaphlan2-f1dcf3958459/db_v20/mpa_v20_m200.pkl --nproc 4 > /home/zgy_ucla_cs/MetaSUB/profile/profiled_SRR1748535.txt



/home/zgy_ucla_cs/tools/bowtie2-2.2.9/bowtie2 -x /home/zgy_ucla_cs/tools/metaphlan2/biobakery-metaphlan2-f1dcf3958459/db_v20/mpa_v20_m200 -f /home/zgy_ucla_cs/MetaSUB/SRR1748535.fastq 
-------




example:
time /home/zgy_ucla_cs/tools/metaphlan2/biobakery-metaphlan2-f1dcf3958459/metaphlan2.py --bowtie2_exe /home/zgy_ucla_cs/tools/bowtie2-2.2.9/bowtie2 --bowtie2out /home/zgy_ucla_cs/temp/SRS014459-Stool.bz2.txt /home/zgy_ucla_cs/temp/SRS014459-Stool.fasta.gz --input_type fasta  > /home/zgy_ucla_cs/temp/result.txt



=========================
Split CMD:
python singleEndToPairedEnd.py --start SRR1748537 --end SRR1748553 --script /home/zgy_ucla_cs/scripts/utility/splitPairedEndReads.pl -v

To do: Finish dsrc; run split on remaining; finishing meta batch 1