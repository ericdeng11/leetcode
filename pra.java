import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

public class pra {
        //This is a file of functions 
	//which atcually mainly implemment a lot of 
	//algorithm and data structure problems
	//Some problems are from leetcode, some are from
	//cracking the code interview
	//some are just basic algs

        public static void helloWorld(){
		System.out.println("Hello World!");
	}
	
	//The solution for partition palindrome list
	//Using Dynamic programming and backtracing technic
	public ArrayList<ArrayList<String>> partition(String s) {
        // Start typing your Java solution below
        // DO NOT write main() function
     	 	  ArrayList<ArrayList<String>> res = new ArrayList<ArrayList<String>>();
      		  ArrayList<String> cur = new ArrayList<String>();
        	int[][] rec = new int[s.length()][s.length()];
        	for( int i = 0 ; i < s.length(); i++ ) rec[i][i] = 1;
        	part(res, cur, 0,rec,s);
        	return res;
    }
    
	public static void part(ArrayList<ArrayList<String>> res, ArrayList<String> cur, int pos, int[][] rec, String s){
       		 if( pos == s.length() ) {
	           ArrayList<String> finalString = new ArrayList<String>();
        	   for( String sub : cur ) finalString.add(sub);
            		res.add(finalString);
            		return;
        }
      	  int max = s.length() - 1 - pos;
      	  for( int i = 0; i <= max; i++){
            if( rec[pos][pos+i] == 0 ){
                if( isPalin(s, pos, pos + i)) rec[pos][pos+i] = 1;
                else rec[pos][pos+i] = -1;
            }
            if( rec[pos][pos+i] == -1 ) continue;
            if( rec[pos][pos+i] == 1) {
                cur.add(s.substring(pos, pos + i + 1));
                part(res, cur, pos + i + 1, rec, s  );
                cur.remove(cur.size() - 1);
            }
            
        }
    }
	static boolean isPalin(String s, int left, int right){
        	if( left >= s.length() || right >= s.length() ) return false;
       		 while( left < right ){
      	      if( s.charAt(left) != s.charAt(right)) return false;
      	      else {
    	            left++;
    	            right--;
    	        }
        }
        return true;
    }
	

	public static void bubbleSort(int[] array){
		int len = array.length;
		for(int i = 0; i < len; i ++){
			for (int j = 1; j < len - i ; j++){
				if(array[j-1] > array[j]){
					int temp = array[j-1];
					array[j-1] = array[j];
					array[j] = temp;
				}
			}
		}
	}
	
	public static void selectSort(int[] array){
		int len = array.length;
		for(int i = 0; i < len; i++){
			int loc = i;
			for(int j = i; j < len; j ++){
				if(array[j] < array[loc]){
					loc = j;
				}
				int temp = array[i];
				array[i] = array[loc];
				array[loc] = temp;
			}
		}
	}
	
	
	public static void mergeSort(int[] a, int low, int high){
		if(low < high){
			int mid = (low + high) / 2;
			mergeSort(a, low, mid);
			mergeSort(a, mid + 1, high);
			merge(a, low, mid, high);
		}
	}
	
	
	public static void merge(int[] a, int low, int mid, int high){
		int[] helper = new int[a.length];
		for(int i = low; i <= high; i ++){
			helper[i] = a[i];
		}
		
		int left = low;
		int right = mid + 1;
		int current = low;
		while(left <= mid && right <= high){
			if(helper[left] <= helper[right] )
			{
				a[current] = helper[left];
				left++;
			}
			else{
				a[current] = helper[right];
				right++;}
			current ++;
		}
		
		if(left <= mid ){
			while(left<=mid){
				a[current] = helper[left];
				current++;
				left++;
			}
		}
		
	}
	
	public static void quickSort(int[] arr, int left, int right){
		int index = partition(arr, left, right);
		if(left< index-1){
		 quickSort(arr, left, index-1);
		}
		if(index < right)
			quickSort(arr, index, right);
	}
	
	public static int partition(int[] arr, int left, int right){
		int key = arr[(left+right)/2];
		while(left <= right ){
			while(arr[left] < key) left++;
			while(arr[right] > key) right--;
			if(left  <= right){
				int temp = arr[left];
				arr[left] = arr[right];
				arr[right] = temp;
				right--;
				left++;
			}
		}
		return left;
	}
	
	
	public static int binarySearch(int[] arr, int key){
		int low = 0;
		int high = arr.length - 1;
		//int mid = (low + high) / 2;
		
		while(low <= high){
			int mid = (low + high) / 2;
			if(arr[mid] < key )
				low = mid + 1;
			else if(arr[mid] > key)
				high = mid - 1;
			else if(arr[mid] == key)
				return mid;
		}
		return Integer.MIN_VALUE;
	}
	
	public static int binaryR(int[] arr, int key, int low, int high){
		if(low > high) return -1;
		int mid = (low + high) / 2;
		if(arr[mid] == key) return mid;
		else if(arr[mid] < key)
			return binaryR(arr, key, mid + 1, high);
		else 
			return binaryR(arr, key, low, mid - 1);
	}
	
	
	
	public static boolean strCheck (String str){
		if(str==null) return false;
		HashMap<Character, Integer> checker = new HashMap<Character, Integer>();
		for(int i = 0; i < str.length(); i ++ ){
			if(checker.containsKey(str.charAt(i)))
					return false;
			else 
			    checker.put(str.charAt(i), 1);
		}
		return true;
	}
	
	
	public static boolean isUnique(String str){
		if(str==null) return false;
		boolean[] set = new boolean[256];
		for(int i = 0 ; i < str.length(); i ++){
			if(set[str.charAt(i)])
				return false;
			else
				set[str.charAt(i)] = true;
		}
		return true;
	}
	
	
	public static String reverse(String str){
		if(str==null) return null;
		String reversed = "";
		for(int i = str.length() - 1; i >= 0; i --)
			reversed += str.charAt(i);
		return reversed;
	}
	
	public static String reverse1(String str){
		if(str==null) return null;
		if(str.length() <= 1)
			return str;
		return reverse1(str.substring(1, str.length())) + str.charAt(0);
	}
	
	public static String moveDup(String str){
		if(str == null) return null;
		if(str.length() == 1) return str;
		
		char[] s_ch = str.toCharArray();
		String uni = "";
		int tail = 1;
		
		for(int i = 1; i < str.length(); i++){
			int j;
			for(j = 0; j < tail ; j++){
				if(s_ch[j] == s_ch[i])
					break;
			}
			
			if(j == tail)
			{
				s_ch[tail] = s_ch[i];
				tail++;
 			}
			
		}
		
		for(int t = 0; t < tail; t++){
			uni += s_ch[t];
		}
		
		return uni;
		
	}
	
	public static void moveDuplicate(char[] str){
		if(str == null) return;
		if(str.length == 1) return;
		int tail = 1;
		
		for(int i = 1; i < str.length; i++){
			int j;
			for(j = 0; j < tail ; j++){
				if(str[j] == str[i])
					break;
			}
			
			if(j == tail)
			{
				str[tail] = str[i];
				tail++;
 			}
			
		}
		
		str[tail] = '0';
		return;
		
	}
	
public static String replace(String str){
	if(str == null) return null;
	String replaced = "";
	for(int i = 0; i < str.length(); i ++){
		if(str.charAt(i) == ' ')
			replaced += "%20";
		else 
			replaced += str.charAt(i);
	}
	return replaced;
}
	
	
public static void replaceFun(char[] str){
	int newLength = 0;
	int len = str.length;
	for(int i = 0; i < len; i++)
		if(str[i] ==' ')
			newLength ++;
	newLength = len + newLength*2;
	
	str[newLength] = '\0';
	for(int i = len-1 ; i > -1; i--){
		if(str[i] == ' ')
		{
			str[newLength-1] = '0';
			str[newLength-2] = '2';
			str[newLength-3] = '%';
			newLength = newLength - 3;
		}
		else {
			str[newLength - 1] = str[i];
			newLength = newLength - 1;
		}
	}
}


// 1.5
public static void swap(int[][] matrix, int n ){
	for(int layer = 0; layer < n/2; layer++){
		int first = layer;
		int last = n-1-layer;
		for(int i = first; i < last; i++){
			int offset = i - first;
			int top = matrix[first][first + offset]; // record the top element
			//left  - > top
			matrix[first][first + offset ] = matrix[last - offset][first];
			//bottom - > left
			matrix[last - offset][first] = matrix[last][last - offset];
			//right - > bottom
			matrix[last][last - offset] = matrix[first + offset][last];
			//top - > right
			matrix[first + offset][last] = top;
		}
	}
}


//2.1 
public static void deleDup(LinkedList l ){
	HashMap<Integer, Integer> rec = new HashMap<Integer, Integer>();
	ListNode pre = l.header;
	ListNode iter = pre.next;
	rec.put(pre.data, 1);
	
	while(iter != null){
		if(rec .containsKey(iter.data)){
			pre.next = iter.next;
			iter = iter.next;
		}
		else{
			rec.put(iter.data, 1);
			pre = iter;
			iter = iter.next;
		}
	}
}

//2.1
public static void deleDup1(LinkedList l ){
	
	ListNode current = l.header.next;
	ListNode pre = l.header;
	ListNode iter;
	
	while(current!= null){
		iter = l.header;
		while(iter != current){
			if(iter.data == current.data){
				pre.next = current.next;
				current = current.next;
				break;
			}
			else{
				iter = iter.next;
			}
		}
		if(iter == current){
		pre = pre.next;
		current = current.next;}
		
	}
}
	
public static void swap (char[] a, char[] b){
	char[] temp = a;
	a = b;
	b = temp;
}


public static boolean getBit (int num, int i ){
	if( i == 31 ) return num < 0;
	return ( (num & (1 << i)) > 0) ;
}

public static int setBit(int num, int i ){
	return num|(1 << i) ;
}
	
public static int clearBit(int num, int i){
	return num & (~(1<<i));
}

public static treeNode insert(int i , int j, int[] array ){
	
	if(j - i < 0) return null;
	treeNode n = new treeNode(array[(j-i)/2+i]);
	n.left = insert(i, (j-i)/2+i-1, array);
	n.right =  insert((j-i)/2+i+1, j, array);
	return n;
	
}

public static treeNode insert1(int start, int end, int[] array){
	if(end < start) return null;
	int mid = (start + end) / 2;
	treeNode n = new treeNode(array[mid]);
	n.left = insert1(start, mid-1, array);
	n.right = insert1(mid+1, end, array);
	return n;
}

public static binaryTree balanceBST(int[] array){
	binaryTree tr = new binaryTree();
	tr.root = insert(0, array.length-1, array);
	return tr;
}

public static boolean isPermutation(String st1, String st2){
	if(st1.length() != st2.length()) return false;
	int rec[] = new int[256];
	for(int i = 0; i < rec.length; i ++){
		rec[i] = 0;
	}
	
	for (int i = 0; i < st1.length(); i ++){
		rec[st1.charAt(i)] ++;
	}
	
	for(int i = 0; i< st2.length(); i++){
		rec[st2.charAt(i)]--;
		if(rec[st2.charAt(i)] < 0) return false;
	}
	
	return true;
}


public static void reverseList(LinkedList l){
	ListNode prev = null;
	ListNode current = l.header;
	ListNode next = current.next;
	while(next != null){
		current.next = prev;
		prev = current;
		current = next;
		next = next.next;
	}
//really weird here	
	current.next = prev;
	l.header = current;
}


public static LinkedList recursiveReverse(ListNode n, LinkedList l){
	
	if(n == null) return l;
	
	l = recursiveReverse(n.next, l);
	l.putNode(n.data);
	return l;
}

	
	public static int[] multiplication(int[] arr){
		int[] res = new int[arr.length];
		int flag = -1;
		int count = 0;
		for(int i = 0; i < arr.length; i++ ){
			if(arr[i] == 0)
			{
				count++;
				flag = i;
			}
		}
		if(count > 1){
			for(int i = 0; i < res.length; i++){
				res[i] = 0;
			}
			return res;
		}
		
		if(count == 1){
			int forward = 1;
			int backward = 1;
			for(int t = 0; t < flag; t++ ){
				forward = forward * arr[t];
			}
			for(int t = arr.length-1; t > flag; t--){
				backward = backward * arr[t];
			}
			for(int t = 0; t < res.length; t++ ){
				if(t == flag )
					res[t] = forward * backward;
				else
					res[t] = 0;
			}
			return res;
		}
		res[0] = 1;
		int backwards = 1;
		for(int i = 1; i < arr.length; i++){
			res[i] = arr[i-1] * res[i-1];
		}
		for(int i = arr.length -1; i>= 0; i--){
			res[i] = res[i] * backwards;
			backwards = backwards * arr[i];
		}
		
		return res;
		
		
	}
	
	
	public static void printCombination(String str){
		char[] in = str.toCharArray();
		int level = 2;
		int len = in.length;
		for(int k = 0; k < len; k++){
			System.out.println(in[k]);
		}
		while(level <= len){
			for(int i = 0; i < len; i++){
				String head = ""+in[i];
				String fi = head;
				
				for(int t = i; t < len; t++){
					for(int j = 1; j < level; j++){
						if(t+j < len){
							fi = fi + in[t+j];
						}
						else break;
					}
					if(fi.length() == level){
					System.out.println(fi);}
			
					fi = head;
				}
			}
			level ++;
		}
	}
	
	
	static void wrapper(String str){
		char[] in = str.toCharArray();
		int len = in.length;
		StringBuffer out = new StringBuffer();
		int level;
		for(level=1; level <= len; level++){
			doPrint(in, out, level, len, 0); 
		}
	}
	
	static void doPrint(char[] in, StringBuffer out, int level, int len, int loc){
		if(out.length() == level ) {
			System.out.println(out.toString());
			return;
		}
		for(int i = loc; i < len; i ++){
			out.append(in[i]);
			doPrint(in, out, level, len, i+1);
			out.setLength(out.length()-1);
		}
	}
	
	static int countCents(int n ){
		int count = 0;
		int total = 0;
		for(int i = 0; i <= n/25 + 1; i++){
			for(int j = 0; j <= n/10 + 1; j++){
				for(int l = 0; l <= n/5 + 1; l ++){
					for(int m = 0; m <= n ; m ++){
						total = i * 25 + j*10 + l*5 + m;
						if(total == n)
						{count++;
						break;}
						if(total > n){
							break;
						}
					}
				}
			}
		}
		return count;
	}
	
	static int Countcents(int sum){
		int[] current = {25, 10, 5, 1};
		int count = recursiveCount(0, 0, current, sum, 0);
		return count;
	}
	
	static int recursiveCount(int count, int total, int[] current, int sum, int loc){
		if(sum == total) return ++count;
		if(sum < total) return count;
		if(loc > current.length-1) return count;
		int currentTotal = total;
		for(int i = 0; i <= sum/current[loc] + 1; i++){
			total += current[loc] * i;
			count = recursiveCount(count, total, current, sum, loc+1);
			total = currentTotal;
		}
		return count;

	}
	//why do i have to use ++loc instead of loc++
	//also why do i have to use ++count instead of count++
	
	
	static void chessQueen(){
		int[] row = {0,1,2,3,4,5,6,7};
		int[] col = {0,0,0,0,0,0,0,0};
		for(int k = 0 ; k < 8; k++){
			col[0] = k;
			chessQueen(row, col, 1);
		}
	}
	
	static void chessQueen(int[] row, int[] col, int level){
		if(level == 8){
			for(int t = 0; t < 8; t++){
					System.out.print(" "+col[t]+ ",");
			}
			System.out.println();
			return;
			
		}
		for(int i = 0; i < 8; i++){
			int flag = -1;
			for(int j = 0; j < level; j++){
			//	System.out.println((Math.abs(col[j]-i )/ Math.abs(row[j] - level)));
				if(col[j] == i) {
				flag = 0; 
				break;
				}
				if((Math.abs(col[j]-i ) == Math.abs(row[j] - level))) {
					flag = 0;
				
				//System.out.println( Math.abs(row[j] - level));
				}
			}
			if(flag == -1){
			col[level] = i;
			row[level] = level;
			chessQueen(row, col, level+1);
			}
		}
	}
	
	
	
	public static int maxHeight(binaryTree t){
		Stack<treeNode> s1 = new Stack<treeNode>();
		Stack<treeNode> s2 = new Stack<treeNode>();
		Stack<treeNode> current = new Stack<treeNode>();
		Stack<treeNode> next = new Stack<treeNode>();
		int maxHeight = 0;
		s1.push(t.root);
		current = s1;
		next = s2;
		while(true){
			while(!current.empty()){
				treeNode n = current.pop();
				System.out.print(n.data + ",");
				if(n.left != null) next.push(n.left);
				if(n.right != null) next.push(n.right);
			}
			System.out.println();
			if(next.empty()) break;
			maxHeight++;
			if(current == s1) 
			{
				current = s2;
				next = s1;
			}
			else{
				current = s1;
				next = s2;
			}
		}
		
		return maxHeight;
	}
	
	public static int maxHeightRecursive(treeNode n){
		if(n == null) return -1;
		return maxHeightRecursive(n.left)>maxHeightRecursive(n.right) ? maxHeightRecursive(n.left) +1: maxHeightRecursive(n.right)+1;
	}
	
	public static void printPrime(int n ){
		if(n < 2) return;
		HashMap<Integer, Integer> rec = new HashMap<Integer, Integer>();
		for(int i = 2; i <= n; i++){
			if(!rec.containsKey(i)) System.out.println(i);
			int non_prime = i;
			int k = 2;
			while(non_prime * k <= n){
				int r = non_prime * k;
				if(!rec.containsKey(r)){
					rec.put(r, 1);
				}
				k++;
			}
		}
 	}
	
	public static void hanoi(int n, String from, String buffer, String to){
		if(n == 1) {
			System.out.println("move "+n+ " from " + from + " to "+buffer);
			System.out.println("move "+n+ " from " + buffer + " to "+to);
			return;
		}
		hanoi(n-1, from, buffer, to);
		System.out.println("move "+n+ " from " + from + " to "+buffer);
		hanoi(n-1, to, buffer, from);
		System.out.println("move "+n+ " from " + buffer + " to "+to);
		hanoi(n-1, from, buffer, to);
		return;	
	}
	
	public static void parentheses(int[] pa, int[] arrange, int level){
		if(level == arrange.length){
			Stack<Integer> check = new Stack<Integer>();
			for(int i = 0; i < arrange.length; i++){
				if(arrange[i] == 0){
					check.push(0);
				}else{
					if(check.isEmpty()) return;
					int c = check.pop();
					if(c == 1) return;
				}
			}
			for(int i = 0; i < arrange.length; i++){
			char c = (arrange[i]==0) ?'(':')';
				System.out.print(c);
			}
			System.out.println();
			return;
		}
		
		
		for(int i = 0; i < 2; i++){
			if(pa[i]!=0){
				arrange[level] = i;
				pa[i]--;
				parentheses(pa, arrange, level+1);
				pa[i]++;
			}
		}
		return;
	}
	
	public static ListNode reverseLi(ListNode n, ListNode oldHead, ListNode newHead){
		if( n.next == null){
			newHead.next= n;
			return n;
		}
		ListNode next = reverseLi(n.next, oldHead, newHead);
		next.next = n;
		if(n != oldHead){
		return n;
		}
		else{
			n.next = null;
			//oldHead = null;
		return newHead;
		}
	}
	
	
	public static int maxSubarray(int[] array){
		int sum = 0;
		int max = 0;
		for(int i = 0; i < array.length; i++){
			sum = sum + array[i];
			if(sum > max){
				max = sum;
				}
			else if(sum < 0){
				sum = 0;
			}
			
		}
		
		return max;
	}
	
	
	public static void sumPath(treeNode n, int sum, int  target, LinkedList path){
		if(n == null) return;
		if(n.data + sum > target) return;
		if(n.data + sum == target) {
			path.putNode(n.data);
			path.printNode();
			path.deleteNode(n.data);
			return;
		}
		sum = sum + n.data;
		path.putNode(n.data);
		sumPath(n.left, sum, target, path);
		sumPath(n.right, sum, target, path);
		path.deleteNode(n.data);
		return;
		
	}
	
	
	public static void preOrderTree(binaryTree tree){
		Stack<treeNode> s = new Stack<treeNode>();
		System.out.println(tree.root.data);
		s.push(tree.root);
		
		while(!s.isEmpty()){
			//treeNode tn = s.peek();
			while(s.peek().left != null){
				System.out.println(s.peek().left.data);
				s.push(s.peek().left);
				
			}
			int flag = 0;		
			while(!s.isEmpty() && flag == 0){
				treeNode top = s.pop();
				if(top.right!= null)
				{
					System.out.println(top.right.data);
					s.push(top.right);
					flag = 1;
				}
			}
			
		}
		
	}
	
	
	public static void preOrder(binaryTree tree){
		Stack<treeNode> s = new Stack<treeNode>();
		s.push(tree.root);
		while(!s.isEmpty()){
			treeNode top = s.pop();
			System.out.println(top.data);
			if(top.right != null) s.push(top.right);
			if(top.left != null) s.push(top.left);
		}
	}
	
	
	public static void inOrderTree(binaryTree tree){
		Stack<treeNode> s = new Stack<treeNode>();
		s.push(tree.root);
		
		while(!s.isEmpty()){
			//treeNode tn = s.peek();
			while(s.peek().left != null){
				s.push(s.peek().left);
			}
			
			int flag = 0;
			
			while(!s.isEmpty() && flag == 0){
				treeNode top = s.pop();
				System.out.println(top.data);
				if(top.right!= null)
				{
					s.push(top.right);
					flag = 1;
				}
			}
			
			
			
		}
		
	}

	
	
	public static ArrayList<int[]> triplets(int[] arr){
		ArrayList<int[]> res = new ArrayList<int[]>();
		quickSort(arr, 0, arr.length - 1);
		for(int i = 0; i < arr.length - 2; i++ ){
			int temp = arr[i];
			int target = 0 - arr[i];
			int left = i + 1;
			int right = arr.length - 1;
			while(left < right){
				if(arr[left] + arr[right] == target){
					int[] r = new int[3];
					r[0] = temp;
					r[1] = arr[left];
					r[2] = arr[right];
					System.out.println(r[0]+" " + r[1]+" " + r[2]);
					res.add(r);
					left++;
					right--;
				}
				else if(arr[left] + arr[right] < target)
					left++;
				else 
					right--;
			}
		}
		return res;
	}
	
	public static void rotateArray(int[] arr, int key){
		if(key >= arr.length)
			key = key % arr.length;
		int left = 0;
		int right = arr.length - 1;
		rotate(arr, left, right);
		left = 0;
		right = key - 1;
		rotate(arr, left, right);
		left = key;
		right = arr.length - 1;
		rotate(arr, left, right);
	}
	
	public static void rotate(int[] arr, int a , int b){
		while(a < b){
			int temp = arr[a];
			arr[a] = arr[b];
			arr[b] = temp;
			a++;
			b--;
		}
	}
	
	
	public static void paintFill(int[][] image, int x, int y, int oriColor, int newColor){
		if(x < 0 || y < 0 || x >= image.length || y >= image[0].length) return;
		if(image[x][y] == newColor || image[x][y] != oriColor) return;
		image[x][y] = newColor;
		paintFill(image, x - 1, y, oriColor, newColor);
		paintFill(image, x , y - 1, oriColor, newColor);
		paintFill(image, x + 1, y, oriColor, newColor);
		paintFill(image, x , y + 1, oriColor, newColor);
	}
	
	public static void paintFillWrapper(int[][] image, int x, int y, int newColor){
		if(x < 0 || y < 0 || x >= image.length || y >= image[0].length) return;
		if(image[x][y] == newColor) return;
		int oriColor = image[x][y];
		paintFill(image, x , y, oriColor, newColor);
	}
	
	
	
	public static void chess(int[] res, int level){
		if(level == 8) {
			for(int i = 0; i < res.length; i++){
				System.out.print(res[i] + "," );
			}
			System.out.println();
			return;
		}
		for( int i = 0; i < 8; i++){
			int flag = 0;
			for(int t = 0; t < level; t++){
				if(res[t] == i) {
					flag = 1;
					break;
				}
				if(Math.abs(level - t) == Math.abs(res[t] - i)){
					flag = 1;
					break;
				}
			}
			if(flag == 0){
				res[level] = i;
				chess(res, level+1);
			}
		}
	}
	
	public static void arrangeArray(int[] arr){
		int len = arr.length;
		int left = 0;
		while(left < len - 1 && arr[left] <= arr[left + 1])
			left++;
		if(left == len - 1){
			System.out.println("Nah");
			return;
		}
		int min = arr[left];
		int minLoc = left;
		for(int i = left + 1; i < len; i++){
			if(arr[i] < min) {
				min = arr[i];
			}
		}
		int it = 0;
		while(it <= left && arr[it] < min)
			it++;
		minLoc = it ;
		
		int right = len - 1;
		while(right > 0 && arr[right] > arr[right - 1])
			right--;
		int max = arr[right];
		int maxLoc = right;
		for(int i = right - 1; i >= 0; i--){
			if(arr[i] > max)
				max = arr[i];
		}
		it = right ;
		while(it < len  && arr[it] < max )
			it++;
		maxLoc = it - 1 ;
		
		System.out.println("(" + minLoc + " " + maxLoc + ")");
	}
	
	
	public static String transferNum (int num, int level, String res, String[] map){
		if(num == 0) return res;
		String partRes = "";
		int smallNum = num % 1000;
		if(smallNum == 0)
			return transferNum( num/1000, level + 1, res, map);
		int hunDig = smallNum / 100;
		if(hunDig != 0)
			partRes = map[hunDig - 1 ] + " hundred ";
		int tenth = smallNum % 100;
		if(tenth < 20 && tenth > 0)
			partRes += map[tenth - 1] + " ";
		else if(tenth >= 20){
			int ten = tenth / 10;
			partRes += map[ten + 17] + " ";
			int dig = tenth % 10;
			if(dig > 0)
				partRes += map[dig - 1] + " ";
		}
		if(level > 0)
			partRes += map[level + 27] + " ";
		res = partRes +", " +res  ;
		return transferNum( num/1000, level + 1, res, map);
	}
	
	public static void numberToString(int num){
		if( num == 0) {
			System.out.println("zero");
			return;
		}
		
		String[] map = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
				"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
				"twenty", "thirty", "fourty", "fifty", "sixty", "seventy", "eighty", "ninety",
				"hundred", "thousand", "million", "billion", "trillion"};
		String res = new String();
		if( num < 0 )
		{
			res = "negative " ;
			num = -1 * num;
		}
		res += transferNum(num, 0, res, map);
		System.out.println(res);
	}
	
	public static int addResIterative ( int a, int b){
		int flag1 = 0;
		if( a < 0 && b < 0) {
			a = -a;
			b = -b;
			flag1 = 1;
		}
		int result = 0;
		int carrier = 0;
		//int result = 0;
		//int i = 0;
		for(int i = 0; i < 31; i++)
		{
			if((getBit(a,i) || getBit(b,i)) == false)
				{
				result = setNewBit(result, i, carrier);
				carrier = 0;
				}
			else if((getBit(a,i) && getBit(b,i)) == true){
				result = setNewBit(result, i, carrier);
				carrier = 1;
			}
			else
			{
				if(carrier == 0){
					result = setNewBit(result, i, 1);
					carrier = 0;
				}
				else{
					result = setNewBit(result, i, 0);
					carrier = 1;
				}
			}
		}
		if(flag1 == 1) result = -result;
		return result;
	}
	
	public static int setNewBit(int n, int loc , int i){
		if(i == 1)  return setBit(n, loc);
		else return clearBit(n, loc);
	}
	
	public static int getOppo(int n){
		if(n == 0 ) return 0;
		int flag1 = 0;
		int flag2 = 0;
		if( n > 0 ) flag1 = 1;
		if( n < 0 ) flag2 = 1;
		if(flag2 == 1) n = n - 1;
		for(int i = 0; i < 32; i++){
			if(getBit(n,i)){
				n = clearBit(n, i);
			}
			else{
				n = setBit(n, i);
			}
		}
		
		if(flag1 == 1 ) n = n + 1;
		return n;
	}
	
	
	public static int findRotate(int key, int[] arr, int left, int right){
		if(left > right ) return Integer.MIN_VALUE;
		int mid = ( right - left ) / 2 + left;
		if(arr[mid] == key) return mid;
		if(arr[left] <= arr[mid]){
			if( arr[mid] > key && arr[left] <= key) return binaryR(arr, key, left, mid - 1 );
			else return findRotate(key, arr, mid + 1, right);
		}
		else {
			if(arr[mid] < key && arr[right] >= key) return binaryR(arr, key, mid + 1, right);
			else return findRotate(key, arr, left, mid - 1);
		}
	}
	
	public static int maxHeight(treeNode n ){
		if(n== null) return 0;
		int left = maxHeight(n.left);
		int right = maxHeight(n.right);
		return Math.max(left, right) + 1;
	}
	
	public static boolean biTree(treeNode n, int max, int min){
		if( n == null ) return true;
		if(n.data < min || n.data > max) return false;
		return biTree(n.left, n.data, min) && biTree(n.right, max, n.data);
	}
	
	public static void dfsWrapper(Tree t){
		int height = maxHeight(t.getRoot());
		for(int i = 0; i < height; i ++){
			dfsTra(t.getRoot(), i, 0);
			System.out.println();
		}
	}
	
	public static void dfsTra(treeNode n, int level, int current){
		if(n == null ) return;
		if( current == level ){
			System.out.print(n.data + ",");
			return;
		}
		dfsTra(n.left, level, current + 1);
		dfsTra(n.right, level, current + 1);
	}
	
	 public ArrayList<ArrayList<Integer>> zigzagLevelOrder(treeNode root) {
	        // Start typing your Java solution below
	        // DO NOT write main() function
	        ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
	        Stack<treeNode> s1 = new Stack<treeNode>();
	        Stack<treeNode> s2 = new Stack<treeNode>();
			s1.push(root);
			while((!s1.isEmpty()) || (!s2.isEmpty())){
	            ArrayList<Integer> temp = new ArrayList<Integer>();
				while(!s1.isEmpty()){
					treeNode n = s1.pop();
	                temp.add(n.data);
					if(n.left != null) s2.push(n.left);
					if(n.right != null) s2.push(n.right);
				}
	            res.add(temp);
	            ArrayList<Integer> temp1 = new ArrayList<Integer>();
				System.out.println();
				while(!s2.isEmpty()){
					treeNode n = s2.pop();
	                temp1.add(n.data);
					if(n.right != null) s1.push(n.right);
					if(n.left != null) s1.push(n.left);
				}
	            res.add(temp1);
			}
	        return res;
	        
	    }
	
	public static void zigzacTree(binaryTree t){
		Stack<treeNode> s1 = new Stack<treeNode>();
		Stack<treeNode> s2 = new Stack<treeNode>();
		s1.push(t.root);
		while((!s1.isEmpty()) || (!s2.isEmpty())){
			while(!s1.isEmpty()){
				treeNode n = s1.pop();
				System.out.print(n.data + ",");
				if(n.left != null) s2.push(n.left);
				if(n.right != null) s2.push(n.right);
			}
			System.out.println();
			while(!s2.isEmpty()){
				treeNode n = s2.pop();
				System.out.print(n.data+",");
				if(n.right != null) s1.push(n.right);
				if(n.left != null) s1.push(n.left);
			}
			System.out.println();
		}
	}
	
	public static void prettyPrint(binaryTree t){
		Queue<treeNode> q1 = new Queue<treeNode>();
		Queue<treeNode> q2 = new Queue<treeNode>();
		int h = maxHeight(t.root);
		int decrease = 1;
		int gap = decrease * 2;
		int armLength = 1;
		int re = h;
		while(re > 1){
			armLength = armLength * 2;
			re--;
		}
		int sp = ((h + 1) * h / 2 )- 1 ;
		q1.enqueue(t.root);
		while(!q1.isEmpty() || !q2.isEmpty()){
			int flag1 = 0;
			int flag2 = 0;
			while(!q1.isEmpty()){
				if(flag1 == 0){
				sp = sp - armLength;
				for(int i = 0; i < sp; i++){
					System.out.print(" ");
				}
				flag1 = 1;
				}
				for(int i = 0; i < armLength; i++)
					System.out.print("-");
				treeNode n = q1.dequeue();
				System.out.print(n.data);
				for(int i = 0; i < armLength; i++)
					System.out.print("-");
				for(int i = 0; i < gap; i++)
					System.out.print(" ");
				if(n.left != null) q2.enqueue(n.left);
				if(n.right != null) q2.enqueue(n.right);
			}
			armLength--;
			System.out.println();
			while(!q2.isEmpty()){
				if(flag2 == 0){
				sp = sp - armLength;
				for(int i = 0; i < sp; i++){
					System.out.print(" ");
				}
				flag2 = 1;
				}
				for(int i = 0; i < armLength; i++)
					System.out.print("-");
				treeNode n = q2.dequeue();
				System.out.print(n.data);
				for(int i = 0; i < armLength; i++)
					System.out.print("-");
				for(int i = 0; i < gap; i++)
					System.out.print(" ");
				if(n.left != null) q1.enqueue(n.left);
				if(n.right != null) q1.enqueue(n.right);
			}
			armLength--;
			System.out.println();
		}
	}
	
	public static boolean saveToFile(binaryTree t) throws IOException{
		if (t == null) return false;
		File file = new File("/Users/Eric/Documents/workspace/practise/binaryTree.txt"); 
		// if file doesn't exists, then create it
		if (!file.exists()) {
			file.createNewFile();
		}
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		toFile(t.root, bw);
		bw.close();
		System.out.println("Done");
		return true;
	}
	
	public static void toFile(treeNode n, BufferedWriter bw) throws IOException{
		if(n == null) return;
		bw.write(n.data + ",");
		toFile(n.left, bw);
		toFile(n.right, bw);
	}
	
	@SuppressWarnings("resource")
	public static binaryTree getTreeBack(String fileName) throws IOException{
		    BufferedReader br = null;
			String sCurrentLine; 
			br = new BufferedReader(new FileReader(fileName));
			StringBuffer sb = new StringBuffer();
			while ((sCurrentLine = br.readLine()) != null) {
				sb.append(sCurrentLine);
			}
			String treeString = sb.toString();
			String[] r = treeString.split(",");
			int[] rec = new int[r.length];
			for(int i = 0; i < r.length; i++)
				rec[i] = Integer.parseInt(r[i]);
			return buildBST(rec);
			
	}
	
	public static binaryTree buildBST(int[] nums){
		binaryTree bt = new binaryTree();
		for(int i : nums)
			bt.insert(i);
		return bt;
	}
	
	public static void seriBiTree(BinaryT t) throws IOException{
		if(t.isEmpty()) return;
		File file = new File("/Users/Eric/Documents/workspace/practise/serilizationBinaryTree.txt"); 
		// if file doesn't exists, then create it
		if (!file.exists()) {
			file.createNewFile();
		}
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		writeToFile(t.getRoot(), bw);
		bw.close();
	}
	public static void writeToFile(treeNode n, BufferedWriter bw) throws IOException{
		if(n == null) return;
		String left;
		String right;
		if(n.left == null ) 
			left = "null";
		else left =String.valueOf(n.left.data);
		if(n.right == null ) 
			right = "null";
		else right =String.valueOf(n.right.data);
		bw.write(n.data +  "," + left + "," + right + "-");
		writeToFile(n.left, bw);
		writeToFile(n.right, bw);
	}
	
	@SuppressWarnings("resource")
	public static BinaryT deseriBiTree(String fileName) throws IOException{
		BufferedReader br = null;
		String sCurrentLine; 
		br = new BufferedReader(new FileReader(fileName));
		StringBuffer sb = new StringBuffer();
		while ((sCurrentLine = br.readLine()) != null) {
			sb.append(sCurrentLine);
		}
		
		String s = sb.toString();
		System.out.println(s);
		String[] firstAttemp = (s.split("-"));
//		for(int i = 0; i < firstAttemp.length; i++){
//			System.out.print(firstAttemp[i] + "\n");
//		}
		ArrayList<String[]> sec = new ArrayList<String[]>();
		for(int i = 0; i < firstAttemp.length; i++){
			String[] temp = firstAttemp[i].split(",");
			sec.add(temp);
		}
		
		return buildBT(sec);
	}
	
	public static BinaryT buildBT(ArrayList<String[]> a){
		BinaryT b = new BinaryT();
		String[] s = a.get(0);
		int t = Integer.parseInt(s[0]);
		b.insert(t);
		if(!s[1].equals("null"))
			b.getRoot().left = new treeNode(Integer.parseInt(s[1]));
		else
			b.getRoot().left = null;
		if(!s[2].equals("null"))
			b.getRoot().right = new treeNode(Integer.parseInt(s[2]));
		else
			b.getRoot().right = null;
		a.remove(0);
		b = buildBT(a, b);
		return b;
	}
	public static BinaryT buildBT(ArrayList<String[]> a, BinaryT b){
		if( a.isEmpty() ) return b;
		String[] s = a.get(0);
		int key = Integer.parseInt(s[0]);
		treeNode n = b.getNode(key);
		if(!s[1].equals("null"))
			n.left = new treeNode(Integer.parseInt(s[1]));
		else
			n.left = null;
		if( !s[2].equals("null") )
			n.right = new treeNode(Integer.parseInt(s[2]));
		else
			n.right = null;
		a.remove(0);
		return buildBT(a, b);
	}
	
	
	public static String excelSheet(int n){
		char[] table = {'a', 'b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
				'r','s','t','u','v','w','x','y','z',};
		String res = new String();
		res = excelSheet(n, res, table);
		return res.toString();
	}
	public static String excelSheet(int n , String res, char[] table){
		if( n < 26 ){
			res = table[n] + res; 
			return res;
		}
		int temp = n % 26 ; 
		res = table[temp] + res;
		return excelSheet(n / 26 - 1, res, table);
	}
	
	public static int[] search2DMartix(int[][] arr, int key, int x1, int y1, int x2, int y2){
		int n = arr.length;
		int m = arr[0].length;
		if(x1 >= n || x1 < 0 || x2 >= n || x2 < 0 || y1 >= m || y1 < 0 || y2 >=m || y2 < 0 ) return null;
		if(x1 > x2 || y1 > y2 ) return null;
		int sx = x1;
		int sy = y1;
		while( sx <= x2 && sy <= y2 && sx < n && sy < m &&arr[sx][sy] < key ){
			sx++;
			sy++;
		}
		int[] result = new int[2];
		if(sx < n && sy < m && arr[sx][sy] == key ){
			result[0] = sx;
			result[1] = sy;
			return result;
		}
		int[] result1 = search2DMartix(arr, key, sx, y1, x2, sy - 1) ;
		int[] result2 = search2DMartix(arr, key, x1, sy, sx - 1, y2);
		return result1 != null ? result1 : result2;
	}
	
	public static int[] search2DMartix(int[][] arr, int key){
		int sx = 0;
		int sy = arr[0].length - 1;
		while( sx < arr.length  && sy >= 0 ){
			if(arr[sx][sy] == key){
				int[] res = new int[2];
				res[0] = sx;
				res[1] = sy;
				return res;
			}
			else if(arr[sx][sy] < key)
				sx++;
			else
				sy--;
		}
		return null;
	}
	
	public static void antiClockwisePrint(BinaryT bt){
		if(bt.isEmpty()) return;
		System.out.println(bt.getRoot().data);
		Queue<treeNode> q1 = new Queue<treeNode>(); 
		Queue<treeNode> q2 = new Queue<treeNode>(); 
		Queue<treeNode> q3 = new Queue<treeNode>(); 
		treeNode iter;
		if(bt.getRoot().left!=null){
		iter = bt.getRoot().left;
		q1.enqueue(iter);
		while(iter != null){
			if(iter.right != null ) q2.enqueue(iter.right);
			if(iter.left != null ) q1.enqueue(iter.left);
			iter = iter.left;
		}
		printQueue(q1);
		System.out.println();
		printQueue1(q2);
		//q2.dequeue();
		//printQueue(q2);
		System.out.print("+++");
		}
		Stack<treeNode> s = new Stack<treeNode>();
		if(bt.getRoot().right != null ){
		treeNode top = bt.getRoot().right;
		while(top != null){
		s.push(top);
		if( top.left != null ) q3.enqueue(top.left);
		top = top.right;
		}
		printQueue1(q3);
		//printQueue(q3);
		System.out.println();
		while(!s.isEmpty()){
			treeNode t = s.pop();
			System.out.print(t.data + ",");
		}
		}
 		
	}
	
	public static void printQueue(Queue<treeNode> q){
		while(!q.isEmpty()){
			treeNode t = q.dequeue();
			System.out.print(t.data + ",");
		}
	}
	public static void printQueue1(Queue<treeNode> q){
		Queue<treeNode> q1 = new Queue<treeNode>(); 
		while(!q.isEmpty()){
			treeNode t = q.dequeue();
			if(t.left == null && t.right == null)
				q1.enqueue(t);
			if( t.left != null ) q.enqueue(t.left);
			if( t.right != null ) q.enqueue(t.right);
		}
		printQueue(q1);
	}
	
	public static ArrayList<ArrayList<int[]>> uniquePath(int[][] graph, int sx, int sy, int fx, int fy){
		ArrayList<ArrayList<int[]>> result = new ArrayList<ArrayList<int[]>>();
		ArrayList<int[]> path = new ArrayList<int[]>();
		result = uniquePath(graph, sx, sy, fx, fy, result, path);
		return result;
	}
	
	public static  ArrayList<ArrayList<int[]>> uniquePath(int[][] graph, int cx, int cy,int fx, int fy, ArrayList<ArrayList<int[]>> result, ArrayList<int[]> path){
		if(cx < 0 || cx >= graph.length || cy < 0 || cy >= graph[0].length){ return result;}
		int[] res = {cx, cy};
		path.add(res);
		if(cx == fx && cy == fy) 
		{
			result.add(path);
			return result;
		}
		ArrayList<int[]> path1 =  new ArrayList<int[]>();
		ArrayList<int[]> path2 =  new ArrayList<int[]>();
		for(int[] a : path ){
			path1.add(a);
			path2.add(a);
		}
		uniquePath(graph, cx + 1, cy, fx, fy, result, path1);
		uniquePath(graph, cx , cy + 1, fx, fy, result, path2);
		return result;
	}
	
	public static int uniquePath1(int[][] graph, int sx, int sy, int fx, int fy){
		int[][] memo = new int[graph.length][graph[0].length];
		for(int i = sx; i <= fx; i++){
			memo[i][sy] = 1;
		}
		for(int j = sy; j <= fy; j++){
			memo[sx][j] = 1;
		}
		for(int i = sx; i <= fx; i++){
			for(int j = sy; j <= fy; j++){
				if( (i - 1 >= 0) && (j - 1 ) >= 0)
					memo[i][j] = memo[i-1][j] + memo[i][j - 1];
			}
		}
		return memo[fx][fy];
		
	}
	
	public static String replaceString(String str, String pattern, char replace)
	{
		char[] s = str.toCharArray();
		char[] p = pattern.toCharArray();
		StringBuffer sb = new StringBuffer();
		int iters = 0;
		while(iters < s.length){
			while(iters < s.length && s[iters] != p[0]){
				sb.append(s[iters]);
				iters++;
			}
			if(iters >= s.length) break;
			int end = findEnd(s, p, iters);
			if( end != -1){
				sb.append(replace);
				iters = end + 1;
			}
			else
			{
				sb.append(s[iters]);
				iters++;
			}
		}
		return sb.toString();
	}
	
	public static int findEnd(char[] s, char[] p, int iters){
		int possibleEnd = 0;
		int iterp = 0;
		while(iters < s.length){
			while(iters < s.length && iterp < p.length && s[iters] == p[iterp]){
				iters++;
				iterp++;
			}
			if( iterp == p.length ){
				possibleEnd = iters - 1;
				iterp = 0;
			}
			else 
				break;
		}
		if( possibleEnd == 0 )
			return -1;
		else 
			return possibleEnd;
	}
	
	public static int maxProfit(int[] price){
		if(price.length < 2) return -1;
		int[] c = new int[price.length - 1];
		int max = 0;
		int sum = 0;
		for(int i = 1; i < price.length ; i++)
			c[i - 1] = price[i] - price[i - 1];
		for(int i = 0; i < c.length ; i++){
			sum = sum + c[i];
			if(sum < 0) sum = 0;
			if(sum > max ) {
	             max = sum;
			}
		}
		return max;
	}
	
	public static int maxProfit1(int[] price){
		if(price.length < 2) return 0;
		int max = 0;
		int sum = 0;
		for(int i = 0; i < price.length - 1; i++){
			sum = sum + price[i + 1] - price[i];
			if( sum < 0) sum = 0;
			if( sum > max ) max = sum;
		}
		return max;
	}
	
	public static int secondLargest(List<Integer> l){
		int largest = Integer.MIN_VALUE;
		int secLargest = Integer.MIN_VALUE;
		for(int i : l){
			if( i >= largest){
				secLargest = largest;
				largest = i;
			}
			else if( i > secLargest ){
				secLargest = i;
			}
		}
		return secLargest;
	}
	
	 public static String strHighLight(String str, String keyword, String openHLTag, String closeHLTag)
	  {
		 StringBuffer sb = new StringBuffer();//cache the final result
		 String[] rec = str.split(" ");       //split the string by space; get all the terms
		 String[] key = keyword.split(" ");   //split the keyword by space
		 for( int i = 0; i < rec.length; i ++){ //Go over the string to find keywords
			 int flag = 0;
			 //if we find the beginning of the keyword in the string
			 //we then look into the next few words in the string and see if they match the keywords
			 if(rec[i].contains(key[0])){
				 int k = 1;
				 //Firstly see if the words have the key words
				 //if not, it's not what we need
				 for(int t = 0; t < key.length; t++ ){
					 if(rec[i + flag].charAt(0) == '<' && !rec[i + flag].contains(">")){
						 while(!rec[i + flag].contains(">")) flag++;
						 //This block handles the situation that there are several words in the tag
						 //so we need to find the end of the tag by searching ">"
					 }
					 if(!rec[i + flag].contains(key[t]))
						 k = -1;
					 //compare the two strings, to see if the two strings have exact the same key chars
					 //can't less or more than the key word
					 String cmp1 = rec[i + flag];
					 String cmp2 = key[t];
					 int iter1 = 0;
					 int iter2 = 0;
					 if(cmp1.charAt(0) == '<'){//If the string starts with a tag
					 while( cmp1.charAt(iter1) != '>'){//find the end of the tag
						 iter1 ++;
					 }
					 }
					 else {
						 //if there are more than one word in the tag
						 //find the end of the tag and compare the word with key word
						 int hasLeft = 0;
						 while(iter1 < cmp1.length() && cmp1.charAt(iter1) != '>'){
							 if(cmp1.charAt(iter1) == '<')
								 hasLeft = -1;
							 iter1++;
						 }
						 if( iter1 == cmp1.length() || hasLeft == -1) iter1 = 0;
					 }
					 if(cmp1.charAt(iter1) == '>') iter1++;
					 while(iter2 < cmp2.length()){
						 if(cmp1.charAt(iter1) != cmp2.charAt(iter2))
							 {k = -1;
							 break;}
						 iter1++;
						 iter2++;
					 }
					 if( !(iter1 == cmp1.length()  || cmp1.charAt(iter1) == '<')){
						 k = -1;
					 }
					 flag ++;
				 }
				 //if the consequent words match with the key words
				 //now we begin to insert tags
				 if( k == 1){
				 int iter = 0;
				 //iter is used to find next word in the string
				 int wordCount = key.length;
				 //word count is used to mark how many words have been completed
				 while( i + iter < rec.length && wordCount > 0){
					 //There are several different situations
					 if(rec[i + iter].charAt(0) == '<' && !rec[i + iter].contains(">")){
						 while(!rec[i + iter].contains(">")) iter++;
						 //This block is used to find the end of a tag
					 }
					 String r = rec[i + iter];
					 //If the doesn't start with "<" and end with ">"
					 //only when it's on the edge of the key words or 
					 //it's left or right word has tags we add tag on this word
					 if(r.charAt(0) != '<' && r.charAt(r.length() - 1) != '>'){
						 if( iter == 0 || rec[i + iter - 1 ].charAt(rec[i + iter - 1 ].length() - 1) == '>')
							 rec[i + iter ] = addLeft(r, openHLTag);
						 if( iter == key.length - 1 || rec[i + iter + 1 ].charAt(0) == '<')
							 rec[i + iter ] = addRight(rec[i + iter ], closeHLTag);
					 }
					 //If the word starts with "<" and doesn't end with ">"
					 //we add tag on the left part
					 else if(r.charAt(0) == '<' && r.charAt(r.length() - 1) != '>') {
						 rec[i + iter ] = addLeft(r, openHLTag);
					 }
					//If the word doesn't start with "<" and ends with ">"
					 //we add tag on the right part
					 //also if the origin tag have more than one words, here we possibly will add
					 //open tag on the left
					 else if(r.charAt(0) != '<' && r.charAt(r.length() - 1) == '>') {
						 int test = 0;
						 while(test < r.length() - 1 && r.charAt(test) != '>'){
							 test++;
						 }
						 if(test < r.length() - 1 && r.charAt(test) == '>') {
							 rec[i + iter ] = addLeft(rec[i + iter ], openHLTag);
							 }
						 
						 rec[i + iter ] = addRight(rec[i + iter ] , closeHLTag);
					 }
					 //if the word begins with "<" and ends with ">"
					 //we add tag on both sides
					 else if(r.charAt(0) == '<' && r.charAt(r.length() - 1) == '>') {
						 rec[i + iter ] = addRight(rec[i + iter ], closeHLTag);
						 rec[i + iter ] = addLeft( rec[i + iter ] , openHLTag);
					 }
					 iter++;
					 wordCount--;
				 }
				 }

			 }
			 }
		 for( String s : rec )
			 {sb.append(s);
			 sb.append(" ");}
		 //return the string buffer
		 return sb.toString();
	  }
	 
	 //Add open tag on the left of the String
	 public static String addLeft(String str, String left){
		 if( str.contains(">")){
		 int iter = 0; 
		 int hasLeft = 0;
		 while(iter < str.length() && str.charAt(iter) != '>'){
			 if(str.charAt(iter) == '<')
				 hasLeft = -1;
			 iter++;
		 }
		 if( iter == str.length() || hasLeft == -1) iter = 0;
	 
		 while(str.charAt(0) == '<' && str.charAt(iter) != '>'){
			 iter ++;
		 }
		 StringBuffer sb = new StringBuffer();
		 char[] rec = str.toCharArray();
		 for(int c = 0; c <= iter ; c++)
			 sb.append(rec[c]);
		 sb.append(left);
		 for(int c = iter + 1; c <= rec.length - 1 ; c++)
			 sb.append(rec[c]);
		 return sb.toString();
		 }
	 
		 else {
			 String res = left + str;
			 return res;
		 }
			
	 }
	 //Add close tag on the left of the String
	 public static String addRight(String str, String right){
		 if(str.charAt(str.length() - 1) == '>'){
		 int iter = str.length() - 1;
		 while( str.charAt(iter) != '<'){
			 iter --;
		 }
		 
		 StringBuffer sb = new StringBuffer();
		 char[] rec = str.toCharArray();
		 for(int c = 0; c < iter ; c++)
			 sb.append(rec[c]);
		 sb.append(right);
		 for(int c = iter ; c < rec.length  ; c++)
			 sb.append(rec[c]);
		 return sb.toString();
		 }
		 else
			 return str + right;
	 }
	 
	 
	 public static void solve(char[][] board) {
	        // Start typing your Java solution below
	        // DO NOT write main() function
	        for( int i = 0; i < board.length; i ++){
	            for(int j = 0; j < board[0].length; j++){
	               if( isCaptured(board, i, j))
	            	   board[i][j] = 'X';
	            }
	        }
	    }
	    
	 public static boolean isCaptured(char[][] board, int x, int y){
	        if( x >= board.length || x < 0 || y >= board[0].length || y < 0) return false;
	        if( board[x][y] == 'X' ) return true;
	        board[x][y] = 'X';
	        if( isCaptured(board, x - 1, y ) &&  isCaptured(board, x + 1, y ) &&  isCaptured(board, x , y -1 ) &&  isCaptured(board, x , y+1 ) ){
	        	board[x][y] = 'O';
	        	return true;
	        }
	        else{
	            board[x][y] = 'O';
	            return false;
	        }
	    }
	
	 
	 public static int minPathSum(int[][] grid) {
	        // Start typing your Java solution below
	        // DO NOT write main() function
	        int lenx = grid.length;
	        int leny = grid[0].length;
	        int[][] res = new int[lenx][leny];
	        res[0][0] = grid[0][0];
	        for( int i = 1 ; i < leny; i++ )
	            res[0][i] = res[0][ i - 1 ] + grid[0][i];
	        for( int i = 1 ; i < lenx; i++ )
	            res[i][0] = res[i - 1][ 0 ] + grid[i][0];
	        for( int i = 1 ; i < lenx; i++){
	            for( int j = 1; j < leny; j++){
	                res[i][j] = Math.min(res[i - 1][j], res[i][ j - 1 ] ) + grid[i][j];
	            }
	        }
	        return res[lenx - 1 ][leny - 1];
	    }
	 
	 public static long mutilSum(int n){
		 if( n < 1 ) return 0;
		 long res = 0;
		 int iter = 1;
		 while( iter < n){
			 if( iter%3 == 0 || iter %5 == 0){
				 res = res + iter;
			 }
			 iter ++;
		 }
		 return res;
	 }
	 
	 public static double findMedianSortedArrays(int A[], int B[]) {
	        // Start typing your Java solution below
	        // DO NOT write main() function
	        if(A == null && B == null ) return -1.0;
	        if( A == null || B == null){
	            return (A == null) ? B[(B.length - 1 )/2] : A[(A.length - 1 )/2] ;
	        }
	        int leftA = 0 ;
	        int leftB = 0;
	        int rightA = A.length - 1;
	        int rightB = B.length - 1;
	        int midA = (rightA + leftA)/2;
	        int midB = (rightB + leftB)/2;
	        while(leftA <= rightA && leftB <= rightB){
	        	
	        midA = (rightA + leftA)/2;
	        midB = (rightB + leftB)/2;
	        System.out.println(midA);
	        System.out.println(midB);
	        if( A[midA] == B[midB] ){
	            return A[midA];
	        }
	        else if( A[midA] > B[midB] ){
	            rightA = midA;
	            leftB = midB + 1;
	        }
	        else if(A[midA] < B[midB] ){
	            leftA = midA + 1 ;
	            rightB = midB;
	        }
	        }
	        
	        return (leftA <= rightA) ? A[midA] : B[midB];
	        
	    }
	 
	 public static int atoi(String str) {
	        // Start typing your Java solution below
	        // DO NOT write main() function
	        if( str == null  ) return 0;
	        if( str.length() == 0 ) return 0;
	        int flag = 1;
	        int start = 0;
	        while(str.charAt(start) == ' ') {start++;}
	        if( str.charAt(start) == '-')  {
	        flag = -1;
	        start++;
	        }
	        if( str.charAt(start) == '+')  {
	        flag = 1;
	        start ++;
	        }
	        int res = 0;
	        for( int i = start; i < str.length(); i ++){
	            if(str.charAt(i) < 48 || str.charAt(i) > 57 ) return res * flag;
	            
	            if(res != 0&&((Integer.MAX_VALUE  - res  )/res  < 10 || Integer.MAX_VALUE - res * 10  < (str.charAt(i) - 48) ) ){
	            	if( flag == 1 )
		                return Integer.MAX_VALUE;
		            else 
		                return Integer.MIN_VALUE;
	            }
	     
	                else res = res*10 + (str.charAt(i) - 48);
	            System.out.println(res);
	            
	        }
	        return res * flag;
	        
	    }
	 public static String longestCommonPrefix(String[] strs) {
	        // Start typing your Java solution below
	        // DO NOT write main() function
	        if( strs == null ) return null;
	        if( strs.length == 0 ) return null;
	        StringBuffer prefix = new StringBuffer(strs[0]);
	        //System.out.println(strs[1].length());
	        for( int i = 1; i < strs.length; i++){
	            int len =  strs[i].length();
	            if( len > prefix.length() ) len = prefix.length();
	            else prefix.setLength(len);
	            for( int j = 0; j < len; j++){
	                if(strs[i].charAt(j) != prefix.charAt(j)){
	                    if( j == 0 ) return null;
	                    else prefix.setLength(j );
	                    break;
	                }
	                
	            }
	        }
	        return prefix.toString();
	    }
	
	
	
	public static void solve1(char[][] board) {
	        // Start typing your Java solution below
	        // DO NOT write main() function
	        if( board == null ) return;
	        if( board.length <= 1 || board[0].length <= 1 ) return; 
	        boolean[][] table = new boolean[board.length][board[0].length];
	        for(int i = 0; i < board[0].length; i ++){
	            if( board[0][i] == 'O' && table[0][i] == false)
	                so(board, 0, i , table);
	        }
	        for(int i = 0; i < board.length; i ++){
	            if( board[i][0] == 'O' && table[i][0] == false)
	                so(board, i, 0 , table);
	        }
	        for(int i = 0; i < board[0 ].length; i ++){
	            if( board[board.length - 1 ][i] == 'O' && table[board.length - 1][i] == false)
	                so(board, board.length - 1, i , table);
	        }
	        for(int i = 0; i < board.length; i ++){
	            if( board[i][board[0].length - 1] == 'O' && table[i][board[0].length - 1] == false)
	                so(board, i, board[0].length - 1 , table);
	        }
	        for( int i = 0; i < board.length; i++){
	            for (int j = 0 ; j < board[0].length - 1; j ++){
	                if(board[i][j] == 'O' && table[i][j] == false){
	                    board[i][j] = 'X';
	                
	            }
	        }
	        
	        
	    }
	    }
	    
	    public static void so(char[][] board, int x, int y, boolean[][] table){
	        if( x < 0 || x >= board.length || y < 0 || y >= board[0].length) return;
	        if( board[x][y] == 'X' )  return;
	        if( table[x][y] == true ) return;
	        table[x][y] = true;
	        so(board, x + 1, y, table);
	        so(board, x , y + 1 , table);
	        so(board, x - 1, y, table);
	        so(board, x , y - 1, table);
	    }
	
	
//Using backtracing alg to solve N Queen problem
	 public static ArrayList<String[]> solveNQueens(int n) {
	        ArrayList<String[]> res = new ArrayList<String[]>();
	        if( n < 1 || ( n > 1  && n < 4 ) ) return null;
	        char[][] board = new char[n][n];
	        for( int i = 0 ; i < n; i++){//Initialize the board
	            for( int j = 0 ; j < n; j++){
	                board[i][j] = '.';
	            }
	         }
	        queen(res, board, 0, n );
	        return res;
	    }
	    
	    public static void queen(ArrayList<String[]> res, char[][] board, int level, int n ){
	        if( level == n  ){//When we have an answer
	            String[] str = new String[n];
	            for(int t = 0 ; t < n; t++){
	                StringBuffer sb = new StringBuffer();
	                for(int i = 0 ; i < n ; i ++){
	                	sb.append(board[t][i]);
	                }
	                str[t] = sb.toString();
	            }
	            res.add(str);
	        }
	      //DFS
	        for( int i = 0; i < n; i ++){
	            int flag = -1;
	            for( int l = 1 ; l <= level ; l++ ){
	                if( board[level - l][i] == 'Q'  )
	                   {
	                       flag = 1;
	                       break;
	                   }
	                if( level - l >= 0 && i - l >=0 && board[level - l][ i - l] == 'Q' ){
	                	flag = 1;
	                       break;
	                  }
	                if(level - l >= 0 &&  i + l < n && board[level - l][i + l] == 'Q'){
	                	flag = 1;
	                	 break;
	                  }
	            }
	            if( flag == -1){
	                board[level][i] = 'Q';
	                queen(res, board, level + 1, n);
	                board[level][i] = '.';
	            }
	        }
	    }	
	
	
	
	
	
	
	public static void main(String[] args) throws IOException{
		//System.out.println(excelSheet(4238923));
		//int[] a = {5};
		//System.out.println(findRotate(5, a, 0, a.length - 1));
		String[] strs ={"aacwef","aawef"};
		System.out.println(longestCommonPrefix(strs));
		
		}
	
	
	
	
	
	
}


/*
ArrayList<ArrayList<int[]>> r = uniquePath(testa, 0, 0, 6, 4);
System.out.println(r.size());
for(ArrayList<int[]> ai : r){
	for(int[] aii : ai){
		System.out.print("(" + aii[0] + "," + aii[1] + ")");
	}
	System.out.println();
}*/

//bt.inorderPrint();
		/*
		seriBiTree(bt);
		BinaryT b = deseriBiTree(new String("/Users/Eric/Documents/workspace/practise/serilizationBinaryTree.txt"));
		dfsWrapper(b);
		
		*/
		//b.printNode();
		//System.out.println(biTree(t.root, Integer.MAX_VALUE, Integer.MIN_VALUE));
		//dfsWrapper(t);
		//zigzacTree(t);
		/* try {
			saveToFile(t);
			binaryTree bt = getTreeBack(new String("/Users/Eric/Documents/workspace/practise/binaryTree.txt"));
			bt.print();
		} catch (IOException e) {
			e.printStackTrace();
		}
		*/
		//prettyPrint(t);






