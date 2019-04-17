using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using IronPython.Hosting;

public class PythonTest : MonoBehaviour {

	// Use this for initialization
	void Start () {
        var engine = Python.CreateEngine();
        var source = engine.CreateScriptSourceFromFile("U:\\CS498_VR\\CS498_Final_Project\\Assets\\PythonTestScript.py");
        var code = source.Compile();
        var scope = code.DefaultScope;
        code.Execute();

        // This will get what the variable test_str holds after the script finishes running
        Debug.Log(scope.GetVariable<string>("test_str"));
    }
	
	// Update is called once per frame
	void Update () {
		
	}
}
