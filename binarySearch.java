import java.util.*;
public class binarySearch{

public static void main(String[] args){
  int[] arr = new int[6];
  arr[0] = 1;
  arr[1] = 7;
  arr[2] = 10;
  arr[3] = 2323;
  arr[4] = 103324;
  arr[5] = 1000000;
  System.out.println(bs(1, arr, 0, arr.length - 1));
  List<Integer> l = new ArrayList<Integer>();
  l.add(1);
  l.add(2);
  System.out.println(l.get(0));
  System.out.println(iterativeBs(2323,arr));
  System.out.println(iterativeBs(1,arr));

}

public static int bs(int key, int[] arr, int left, int right){
  if( left > right ) return -1;
  int mid = (left  + right ) / 2;
  if (arr[mid] == key ) return mid;
  else if ( arr[mid] < key ) return bs(key, arr, left + 1, right);
  else return bs(key, arr, left, mid -1);
}

public static int iterativeBs(int key, int[] arr){
     int left = 0;
     int right = arr.length - 1; 
     while( left <= right )   
     {
        int mid = (left + right) / 2;
        if(arr[mid] == key ) return mid;
        else if(arr[mid] < key ) {
          left = mid + 1;   
        }
        else right = mid  - 1;
     } 
     return -1;
}


}



