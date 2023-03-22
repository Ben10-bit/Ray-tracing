# Ray-tracing   
通过本章学习了常量内存的使用以及通过事件来测量程序的性能

1. 常量内存：当半个线程束(16个线程)都读取相同地址时，将该地址设为常量地址将大幅提升性能  
定义方法：`__constant__`  
传输数据采用函数：`cudaMemcpyToSymbol()`

2. 事件：本质上为GPU的时间戳
定义方法：`cudaEvent_t start; cudaEventCreate(&start);`
记录当前时间点：`cudaEventRecord(start, 0);`
事件同步函数：`cudaEventSynchronize(stop)`, 在stop之前的语句未执行之前阻塞后面的语句。
计算两个事件之间经历的时间：`cudaEventElapsedTime(&elapsedTime, start, stop)`
销毁事件：`cudaEventDestroy()`
