from py_file.py_run_cmd import run_cmd
rc = run_cmd('./cache/')
rc.close_pid(rc.find_cmd_pid('ollama'))
rc.close_pid(rc.find_cmd_pid('ollama app'))
rc.close_pid(rc.find_port_pid(9880))
rc.close_pid(rc.find_port_pid(9337))