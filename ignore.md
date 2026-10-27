(.venv-cuda) fastlab@gup-admin-Precision-3680:~/deepFakeDetection/DeepDocForgery$ nano rescue_epoch33_now.py  
(.venv-cuda) fastlab@gup-admin-Precision-3680:~/deepFakeDetection/DeepDocForgery$ sudo gdb -q -n -batch -p 9090 \
  -ex 'set pagination off' \
  -ex 'thread 1' \
  -ex 'call (int) PyRun_SimpleString("exec(open(\"rescue_epoch33_now.py\").read())")' \
  -ex detach
[New LWP 9919]
[New LWP 9847]
[New LWP 9255]
[New LWP 9254]
[New LWP 9253]
[New LWP 9252]
[New LWP 9251]
[New LWP 9250]
[New LWP 9249]
[New LWP 9248]
[New LWP 9247]
[New LWP 9246]
[New LWP 9245]
[New LWP 9244]
[New LWP 9243]
[New LWP 9242]
[New LWP 9241]
[New LWP 9240]
[New LWP 9239]
[New LWP 9238]
[New LWP 9237]
[New LWP 9236]
[New LWP 9235]
[New LWP 9234]
[New LWP 9233]
[New LWP 9232]
[New LWP 9231]
[New LWP 9230]
[New LWP 9229]
[New LWP 9182]
[New LWP 9181]
[New LWP 9180]
[New LWP 9179]
[New LWP 9178]
[New LWP 9177]
[New LWP 9176]
[New LWP 9175]
[New LWP 9174]
[New LWP 9173]
[New LWP 9172]
[New LWP 9171]
[New LWP 9170]
[New LWP 9169]
[New LWP 9168]
[New LWP 9167]
[New LWP 9166]
[New LWP 9165]
[New LWP 9164]
[New LWP 9163]
[New LWP 9162]
[New LWP 9161]
[New LWP 9160]
[New LWP 9159]
[New LWP 9158]
[New LWP 9157]
[New LWP 9156]
[New LWP 9155]
[New LWP 9154]
[New LWP 9150]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
0x00000000005d8fd3 in _PyEval_EvalFrameDefault ()
[Switching to thread 1 (Thread 0x780bdd7b0740 (LWP 9090))]
#0  0x00000000005d8fd3 in _PyEval_EvalFrameDefault ()
$1 = 0
[Inferior 1 (process 9090) detached]
(.venv-cuda) fastlab@gup-admin-Precision-3680:~/deepFakeDetection/DeepDocForgery$ ls -lh output/model/cuda-full/epoch33_EMERGENCY.pt
-rwxrwxrwx 1 fastlab fastlab 590M Oct 27 12:17 output/model/cuda-full/epoch33_EMERGENCY.pt
(.venv-cuda) fastlab@gup-admin-Precision-3680:~/deepFakeDetection/DeepDocForgery$ cat output/model/cuda-full/epoch33_RESCUE_SUCCESS.txt
Saved epoch 33
Checkpoint: output/model/cuda-full/epoch33_EMERGENCY.pt
(.venv-cuda) fastlab@gup-admin-Precision-3680:~/deepFakeDetection/DeepDocForgery$ cat output/model/cuda-full/epoch33_RESCUE_ERROR.txt 2>/dev/null

(.venv-cuda) fastlab@gup-admin-Precision-3680:~/deepFakeDetection/DeepDocForgery$ 