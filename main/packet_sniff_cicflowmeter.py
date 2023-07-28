import subprocess
import time
import shutil
import os
import psutil

def initiate_shell_script(send_ip, dst_ip, filename, interface):
    shell_script_path = "/home/kali/packetsniff.sh"
    # p = subprocess.Popen([shell_script_path, send_ip, dst_ip, filename, interface])
    p = subprocess.Popen(' '.join([shell_script_path, send_ip, dst_ip, filename, interface]), stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    return p
    
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

def run_cicflowmeter(filename):
    cicflowmeter_path = "/home/kali/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425/TCPDUMP_and_CICFlowMeter/CICFlowMeters/CICFlowMeter-4.0/bin/CICFlowMeter"
    subprocess.run([cicflowmeter_path])

def stop_process(p):
    p.terminate()

def main():
    s_ip = "192.168.177.129"
    d_ip = "192.168.177.135"
    filename = "test.pcap"
    interface = "eth0"
    time_to_sniff = 1
    p = initiate_shell_script(s_ip, d_ip, filename, interface)
    time.sleep(time_to_sniff)  
    stop_process(p)
    kill(p.pid)
    if p.poll() is None:  # check if it has terminated
        print("The script didn't terminate as expected.")
    else:
    	print("Finished capturing the packets, proceeeding to extract features using CICFlowMeter.")
    shutil.move(filename, "/home/kali/Desktop/data/in/" + filename)
    time.sleep(10)  
    run_cicflowmeter(filename)
    
if __name__ == "__main__":
    main()