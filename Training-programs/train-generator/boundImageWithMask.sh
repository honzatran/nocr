#!/bin/bash

dirImage="$1";
dirMask="$2";


function getFileName
{
    echo "$1" | sed "s/.*\/\([^/]\)/\1/";
}

find "$dirImage" -type f -print -name "*.jpg" -o -name "*.png" > tmpIm;
find "$dirMask" -type f -print -name "*.jpg" -o -name "*.png" > tmpMsk;

cat tmpIm | 
  while read line; do
      filename=`getFileName "$line" `;
      maskfile=`cat tmpMsk| grep "$filename"`;
      if [ -n  "$maskfile" ]; then
          echo "$line:$maskfile";
      fi;
  done;
     
rm tmpIm tmpMsk;
