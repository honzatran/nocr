#!/bin/bash
input="$1";
tr="$2";
fl="$3"
bad="$3";
msg="[t/f/b] ?"

#function recursiveImageList
#{
#  find -type f  -name "*.jpg" -o -name "*.png"
#}

function readAnswer 
{
    ans="";
    while [ "$ans" != "t" ] && [ "$ans" != "f" ] && [ "$ans" != "b" ] ; do
       echo "$msg" > /dev/tty;
       read ans < /dev/tty;
    done;
    echo "$ans"
}

TMPIFS="$IFS";
IFS=":";

cat "$input"| 
while read image mask; do
  ../mainProject/build/projekt "$image" -tM -m "$mask" ;
  ../mainProject/build/projekt "$image" -tM --invert -m "$mask";
  ans=`readAnswer`;
  case "$ans" in
    "t")
      echo "$image:$mask" >> "$tr";
      ;;
    "f")
      echo "$image:$mask" >> "$fl";
      ;;
    "b")
      echo "$image:$mask" >> "$bad";
      ;;
  esac
done;
IFS="$TMPIFS";
        



