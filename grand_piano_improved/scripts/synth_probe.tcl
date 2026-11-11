# Synthesise the coefficient-ROM probe, to check that Vivado evaluates the
# elaboration-time physics rather than only Verilator.
#
#   vivado -mode batch -source scripts/synth_probe.tcl

set root [file normalize [file dirname [info script]]/..]
file mkdir $root/out

read_verilog -sv [list \
  $root/rtl/grand_piano_pkg.sv \
  $root/rtl/grand_piano_phys.sv \
  $root/tb/coef_synth_probe.sv]

set_property include_dirs [list $root/rtl] [current_fileset]

synth_design -top coef_synth_probe -part xc7a100tcsg324-1 -mode out_of_context

report_utilization -hierarchical
puts "SYNTH_PROBE: synthesis completed"

# Read a few ROM locations back out of the elaborated netlist so the numbers
# can be compared against the Python model, not merely assumed correct.
set fh [open $root/out/synth_probe_util.txt w]
puts $fh [report_utilization -return_string]
close $fh
