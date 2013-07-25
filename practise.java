import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

//import pra.LNode;

class LNode{
	int val;
	LNode next;
	public LNode( int x ){
		val = x;
		next = null;
	}
}


public class practise {
	//Test string, int, char 
	public static void main(String[] args){
	
		
	}
	
	//Solve number permutations
	//Only distinc numbers
public ArrayList<ArrayList<Integer>> permute(int[] num) {
        // Start typing your Java solution below
        // DO NOT write main() function
        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
        int n = num.length;
        ArrayList<Integer> permutation = new ArrayList<Integer>();
        perm(num, n, res, permutation);
        return res;
    }
    
    
public static void perm(int[] num, int n , ArrayList<ArrayList<Integer>> res, ArrayList<Integer> permutation){
        if( permutation.size() == n ){
            ArrayList<Integer> r = new ArrayList<Integer>();
            for( int i : permutation ) r.add(num[i]);
            res.add(r);
            return;
        }
        for( int i = 0; i < n; i++){
            if( permutation.contains(i) ) continue;
            else {
                permutation.add(i);
                perm(num, n, res, permutation);
                permutation.remove(permutation.size() - 1);
            }
        }
    }

//Only return the number of solutions of N Queen Problem
public int totalNQueens(int n) {
      	    if( n < 1 || ( n > 1  && n < 4 ) ) return 0;
            int[] chess = new int[n];
	        int count = queen(chess, 0, n , 0);
	        return count;
	    }
//Use only one array to solve the problem
public static int queen(int[] chess, int level, int n ,int count){
	        if( level == n ) 
                return ++count;
            for( int i = 0 ; i < n; i++){
                int flag = 1;
                for( int j = 1; j <= level; j++){
                    if( chess[level - j] == i || j == Math.abs( chess[level - j] - i)) {
                        flag = 0;
                        break;
                    }               
                }
                if( flag == 1 ){
                    chess[level] = i;
                    count = queen(chess, level + 1, n, count);
                }
                            
            }
            return count;
            
	    }

	
	
	public static LNode switchNode( LNode head ){
		//if( k <= 1 ) return head;
		if( head == null || head.next == null ) return head;
		LNode pre = null;
		LNode cur = head;
		LNode pos = head.next;
		LNode newHead = null;
		while( pos != null ){
			if( pre != null ){
				pre.next = pos;
			}
			else{
				newHead = head.next;
			}
				cur.next = pos.next;
				pos.next = cur;
				pre = cur;
				cur = cur.next;
				if( cur != null ){
				pos = cur.next;
				}
				else pos = null;
		}
		return newHead;
	}
	
	public static ArrayList<String> getString( ArrayList<Character> c, int k){
		ArrayList<String> res = new ArrayList<String>();
		StringBuffer sb = new StringBuffer();
		findKString(c, k, res, sb);
		return res;
	}
	
	public static void findKString( ArrayList<Character> c, int k, ArrayList<String> res, StringBuffer sb){
		if( k == 0 ) {
			res.add(sb.toString());
			return;
		}
		for( int i = 0 ; i < c.size(); i++){
			sb.append(c.get(i));
			findKString(c, k - 1, res, sb);
			sb.setLength(sb.length() - 1);
		}
	}

	
	
	static public boolean isPalindrome(int x) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if( x < 0 ) return false;
        int tes = 10;
        while( x / tes > 10){
            tes = tes * 10;
        }
        return count(x, tes);
        
        
    }
    
   static  boolean count (int x , int fac)
    {
        if( x >= 0 && x < 10  ) return true;
        if( x / fac != x % 10 ) return false;
        return count( ( x % fac ) / 10 , fac / 100);
    }
	
	
	
	
	static public int threeSumClosest(int[] num, int target) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if( num.length < 3 ) return -1;
        Arrays.sort(num);
        int min = Integer.MAX_VALUE;
        int sum = 0;
        for( int i = 0 ; i < num.length; i++){
            int l = 0;
            int r = num.length - 1;
            int gap = 0;
            while( l < r ){
                if( l == i ) l++;
                else if( r == i ) r--;
                else {
                gap = target - num[i] - num[l] - num[r];
                
                if( Math.abs(gap) < min )
                {
                min = Math.abs(gap);
                sum = num[i] + num[l] + num[r];
                }
                if( gap < 0 ) r--;
                else if( gap > 0 ) l++;
                else break;
                }
            }
        }
        return sum;
    }
	
	static public boolean isMatch(String s, String p) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if( s == null || p == null ) return false;
        int iters = 0;
        int iterp = 0;
        while( iters < s.length() && iterp < p.length()){
            char c1 = s.charAt(iters);
            char c2 = p.charAt(iterp);
            if( c1 == c2 ){
                iters++;
                iterp++;
            }
            else if( c1 != c2 && c2 != '*' && c2 != '.'){
                if( iterp == p.length() - 1 ) return false;
                char c3 = p.charAt( iterp + 1 );
                if( c3 != '*' ) return false;
                //iters++;
                iterp += 2 ;
            }
            else if ( c1 != c2 && c2 == '.' ){
                iters++;
                iterp++;
            }
            else if( c1 != c2 && c2 == '*' ){
                if( iterp == 0 ) return false;
                char c4 = p.charAt(iterp - 1);
                if( c1 == c4 || ( c4 == '.' && s.charAt(iters - 1 ) == c1 )){
                    iters++;
                }
                else {
                	//Set<ArrayList<Integer>> set = new HashSet<ArrayList<Integer>>();
                   // iters++;
                    iterp++;
                }
            }
        }
        if( iters < s.length() ) return false;
        while( iterp < p.length() )
        {
            if( p.charAt(iterp) == '*' ) iterp++;
            else if( p.charAt(iterp) != '*' ){
            if( iterp + 1 >= p.length() ) return false;
            if( p.charAt(iterp + 1 ) != '*' ) return false;
            iterp += 2 ;
            } //&& (iterp + 1 > p.length() || p.charAt(iterp + 1 ) != '*' )) return false;
        }
        return true;
    }
	
	static double findMedian(int A[], int B[], int l, int r, int nA, int nB) {
		if (l > r) return findMedian(B, A, Math.max(0, (nA+nB)/2-nA), Math.min(nB, (nA+nB)/2), nB, nA);
		int i = (l+r)/2;
		int j = (nA+nB)/2 - i - 1;
		if (j >= 0 && A[i] < B[j]) return findMedian(A, B, i+1, r, nA, nB);
		else if (j < nB-1 && A[i] > B[j+1]) return findMedian(A, B, l, i-1, nA, nB);
		else {
		if ( (nA+nB)%2 == 1 ) return A[i];
		else if (i > 0) return (A[i]+Math.max(B[j], A[i-1]))/2.0;
		else return (A[i]+B[j])/2.0;
		}
		}
	
	public static LNode partition(LNode head, int x) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if( head == null ) return null;
        LNode faceHead = new LNode(0);
        LNode fackHead = new LNode(0);
        LNode small = faceHead ;
        LNode big = fackHead ;
        LNode tracesm = small;
        LNode tracebig = big;
        LNode iter = head;
        while( iter!= null ){
            if( iter.val < x ) {
                tracesm.next = iter;
                tracesm = tracesm.next;
            }
            else
                {
                    tracebig.next = iter;
                    tracebig = tracebig.next;
                }
                iter = iter.next;
        }
        small = small.next;
        big = big.next;
       if( small == null ) return big;
        if( big == null ) return small;
        tracebig.next = null;
        
        tracesm.next = big;
        return small;
    }
	
	public static int removeDuplicates(int[] A) {
        // Start typing your Java solution below
        // DO NOT write main() function
        if( A == null ) return 0;
        if( A.length == 0 ) return 0;
        List<Integer> l = new ArrayList<Integer>();
        int counter = 1;
        l.add(A[0]);
        for( int i = 1 ; i < A.length; i++){
            if( A[i] == A[i - 1] && counter == 2 ) continue;
            else if(A[i] == A[i - 1] && counter == 1 ){
                l.add(A[i]);
                counter++;
            }
            else{
                l.add(A[i]);
                counter = 1;
                
            }
        }
        int[] a = new int[l.size()];
        for(int i = 0; i < a.length; i++){
            a[i] = l.get(i);
        }
        A = a;
        return l.size();
        
    }
	
	
	
	static void quickSort(int[] s, int left, int right){
        int part = sort(s, left, right);
        if( part + 1 < right)
            quickSort( s, part + 1 , right);
        if( part > left )
            quickSort( s, left, part);
    }
    
    static int sort( int[] s, int left, int right){
        if( left >= right ) return left;
        int l = left;
        int r = right;
        int key = s[left];
        while( l <= r ){
        while( l <= right && s[l] < key ) l++;
        while( r >= left && s[r] > key ) r--;
        if( l > r ) return l - 1 ; 
        swap(s, l, r);
        l++;
        r--;
        }
        return l - 1 ;
        
    }
    static void swap ( int[] s, int x, int y )
{
    int temp = s[x];
    s[x] = s[y];
    s[y] = temp;
}
	
    
    static void mergeSort(int[] s, int left, int right){
    	if( left >= right ) return;
    	int mid = (right + left) / 2 ;
    	mergeSort(s, left, mid  );
    	mergeSort(s, mid + 1 , right);
    	merge(s, left, mid, right);
    }
    
    static void merge(int[] s, int left, int mid, int right){
    	int[] t = new int[s.length];
    	for( int i = left; i <= right; i++){
    		t[i] = s[i];
    	}
    	int l = left;
    	int m = mid + 1;
    	int count = left;
    	while( l <= mid && m <= right ){
    		if( t[l] < t[m]){
    			s[count] = t[l];
    			count++;
    			l++;
    	}
    		else{
    			
    		s[count] = t[m];
    		count++;
    		m++;
    		}
    		}
    	if( l <= mid){
    		while( l <= mid){
    			s[count] = t[l];
    			count++;
    			l++;
    		}
    	}
    }
	
	

}







class Heap{
	private int HEAP_SIZE;
	private int[] heap;
	private int currentSize;
	
	public Heap(int size){
		HEAP_SIZE = size;
		currentSize = 0;
		heap = new int[HEAP_SIZE];
	}
	
	public void insert(int data){
		if(currentSize < HEAP_SIZE){
			heap[currentSize] = data;
			moveUp(currentSize);
			currentSize ++;
		}	
		else if( currentSize == HEAP_SIZE){
			if(data > heap[0] ){
				heap[0] = data;
				moveDown(0);
			}
		}

	}
	
	public void moveUp(int location){
		int btm = heap[location];
		int index = location;
		int parent = (index - 1) /2;
		while(btm < heap[parent] && index > 0){
			heap[index] = heap[parent];
			index = parent;
			parent = (index - 1) / 2;
		}
		heap[index] = btm;			
	}
	
	public void moveDown(int location){
		int smallerChild;
		int top = heap[location];
		int index = location;
		while(index < currentSize / 2){
			int leftChild = 2 * index + 1;
			int rightChild = 2 * index + 2;
			if(rightChild < currentSize && heap[leftChild] > heap[rightChild])
				smallerChild = rightChild;
			else
				smallerChild = leftChild;
			if( top < heap[smallerChild])
				break;
			heap[index] = heap[smallerChild];
			index = smallerChild;
		}
		heap[index] = top;
	}
	
	public void printData(){
		for(int i = 0; i < HEAP_SIZE; i++){
			System.out.println(heap[i]);
		}
	}
	
	private class Entry<K, V>{
		private K key;
		private V value;
		public Entry(K k, V v){
			key = k;
			value = v;
		}
	}
	
	class SortedList<K, V>{
		private Entry<K, V> e;
	}
	
	
	
	
	}


