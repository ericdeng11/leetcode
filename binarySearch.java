import java.util.*;
public class binarySearch{

public static void main(String[] args){
  int[] arr = new int[5];
  arr[0] = 1;
  arr[1] = 7;
  arr[2] = 10;
  arr[3] = 2323;
  arr[4] = 103324;

  System.out.println(bs(1, arr, 0, arr.length - 1));
}

public static int bs(int key, int[] arr, int left, int right){
  if( left > right ) return -1;
  int mid = (left  + right ) / 2;
  if (arr[mid] == key ) return mid;
  else if ( arr[mid] < key ) return bs(key, arr, left + 1, right);
  else return bs(key, arr, left, mid -1);
}

}



