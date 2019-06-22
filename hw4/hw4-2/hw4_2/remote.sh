#!/bin/bash
work_station="b05901027@140.112.48.127"
port=11000
if [ "$#" == "0" ];then
    echo -e "Actions: link load_viz\n"
    exit 0
fi
if [ "${1}" == "link" ];then
    ssh ${work_station} -p $port
fi
if [ "${1}" == "load_viz" ];then
    scp -r -P ${port} ${work_station}:fcn/pytorch-fcn/examples/voc/logs/${2}/visualization_viz/ .
fi
if [ "${1}" == "load" ];then
    scp -r -P ${port} ${work_station}:MLDS/HW4-2/${2} .
fi
if [ "${1}" == "load_model" ];then
    scp -r -P ${port} ${work_station}:MLDS/HW4-2/models/${2} models/${2}
fi
