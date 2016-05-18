
for i in {0..9}
do
	echo $i;
	python generate_input.py $i;
	../../build/tools/caffe train -solver solver_vgg.prototxt -weights VGG_ILSVRC_16_layers.caffemodel 2>&1 | tee log$i
done
