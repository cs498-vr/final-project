using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RayCastTest : MonoBehaviour {
    public LayerMask layerMask;
    public GameObject rightHand;
    public LineRenderer lr;
    public GameObject drawPlane;

    Vector3 lastPoint = new Vector3(3000, 3000, 3000);
    List<Vector3> points = new List<Vector3>();

	// Use this for initialization
	void Start () {
        lr = GetComponent<LineRenderer>();
	}
	
	// Update is called once per frame
	void Update () {
        if (OVRInput.Get(OVRInput.Button.One))
        {
            print("button was pressed");
            RaycastHit hit;
            Vector3 fwd = rightHand.transform.TransformDirection(Vector3.forward);
            if (Physics.Raycast(rightHand.transform.position, fwd, out hit, 20f, layerMask))
            {
                if (Vector3.Magnitude(hit.point - lastPoint) > 0.1)
                {
                    points.Add(Quaternion.Inverse(drawPlane.transform.rotation) * (hit.point - drawPlane.transform.position));
                }
                print("hitting plane");
                Debug.DrawRay(rightHand.transform.position, fwd);
                lr.enabled = true;
                lr.SetPosition(0, rightHand.transform.position);
                lr.SetPosition(1, hit.point);
            }
            else
            {
                lr.enabled = false;
            }
        }
        else
        {
            lr.enabled = false;
            points = new List<Vector3>();
        }

        for (int i = 1; i < points.Count; i++)
        {
            Debug.DrawLine(drawPlane.transform.rotation * points[i - 1] + drawPlane.transform.position,
                drawPlane.transform.rotation * points[i] + drawPlane.transform.position);
        }
    }
}
