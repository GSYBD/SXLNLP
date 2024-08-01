#include<bits/stdc++.h>
using namespace std;

string str[7]={"经","常","有","意","见","分","歧"};
set<string> s;
string word[12]={"经常","经","有","常","有意见","歧","意见","分歧","见","意","见分歧","分"};
string result[10];
vector<vector<string>> ans;

void dfs(int idx,int cnt){
    if(idx==7){
        vector<string> new_ans;
        for(int i=0;i<cnt;++i){
            new_ans.push_back(result[i]);
        }
        ans.push_back(new_ans);
        return;
    }
    string tmp="";
    for(int ed=idx;ed<7;++ed){
        tmp+=str[ed];
        if(s.count(tmp)){
            result[cnt]=tmp;
            dfs(ed+1,cnt+1);
            result[cnt]="";
        }
    }
}

int main(){
    int n;
    cin>>n;
	for(int i=0;i<12;++i){
        s.insert(word[i]);
    }
    dfs(0,0);
    for(int i=0;i<ans.size();++i){
        for(int j=0;j<ans[i].size();++j){
            cout<<ans[i][j];
            if(j<ans[i].size()-1){
                cout<<' ';
            }else{
                cout<<endl;
            }
        }
    }
	return 0;
} 
// 经 常 有 意 见 分 歧
// 经 常 有 意 见 分歧
// 经 常 有 意 见分歧
// 经 常 有 意见 分 歧
// 经 常 有 意见 分歧
// 经 常 有意见 分 歧
// 经 常 有意见 分歧
// 经常 有 意 见 分 歧
// 经常 有 意 见 分歧
// 经常 有 意 见分歧
// 经常 有 意见 分 歧
// 经常 有 意见 分歧
// 经常 有意见 分 歧
// 经常 有意见 分歧