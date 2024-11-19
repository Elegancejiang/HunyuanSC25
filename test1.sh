input="graph1.csv"

{
	read
	i=1
	while IFS=',' read -r  Name 
	do

	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240401_8.txt
	
	# ./cuMetis /media/hemeng/2TB1/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuGP/20240306_8.txt
	# ./cuMetis /media/hemeng/2TB1/graph/$Name.graph 32 >> /home/hemeng/Documents/yongjiang/cuGP/20240306_32.txt
	# ./cuMetis /media/hemeng/2TB1/graph/$Name.graph 64 >> /home/hemeng/Documents/yongjiang/cuGP/20240306_64.txt
	# ./cuMetis /media/hemeng/2TB1/graph/$Name.graph 128 >> /home/hemeng/Documents/yongjiang/cuGP/20240306_128.txt
    # ./cuMetis /media/hemeng/2TB1/graph/$Name.graph 256 >> /home/hemeng/Documents/yongjiang/cuGP/20240306_256.txt
	# ./cuMetis /media/hemeng/2TB1/graph/$Name.graph 512 >> /home/hemeng/Documents/yongjiang/cuGP/20240306_512.txt
	# ./cuMetis /media/hemeng/2TB1/graph/$Name.graph 1024 >> /home/hemeng/Documents/yongjiang/cuGP/20240306_1024.txt

		# ./cuMetis /home/hemeng/Documents/yongjiang/graph/$Name.graph 8 >> /home/hemeng/Documents/yongjiang/cuMetis/20231212.txt

		# ./cuMetis /home/hemeng/Documents/yongjiang/graph/$Name.graph 32 >> /home/hemeng/Documents/yongjiang/cuMetis/ppt.txt
		# ./cuMetis /home/hemeng/Documents/yongjiang/graph/$Name.graph 64 >> /home/hemeng/Documents/yongjiang/cuMetis/ppt.txt
		# ./cuMetis /home/hemeng/Documents/yongjiang/graph/$Name.graph 128 >> /home/hemeng/Documents/yongjiang/cuMetis/ppt.txt
		# ./cuMetis /home/hemeng/Documents/yongjiang/graph/$Name.graph 256 >> /home/hemeng/Documents/yongjiang/cuMetis/ppt.txt

		i=`expr $i + 1`
		done 
} < "$input"
