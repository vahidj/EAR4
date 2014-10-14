/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    EAR4.java
 *    Copyright (C) 2014 Indiana University
 *
 */

package weka.classifiers.lazy;

import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.FastVector;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.AdditionalMeasureProducer;

import java.util.Enumeration;
import java.util.Vector;
import java.util.Collections;
import java.util.List;

/**
 <!-- globalinfo-start -->
 * Ensembles of Adaptations for Regression.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * V. Jalali, D. Leake (2013). Extending Case Adaptation with Automatically-Generated Ensembles of Adaptation Rules. ICCBR 2013: 188-202 
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{jalali2013,
 *    author = {V. Jalali and D. Leake},
 *    title={Extending case adaptation with automatically-generated ensembles of adaptation rules},
 *    booktitle={Case-Based Reasoning Research and Development},
 *    pages={188--202},
 *    year={2013},
 *    publisher={Springer}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -I
 *  Weight neighbours by the inverse of their distance (distance weighting is not supported yet)
 *  (use when k &gt; 1)</pre>
 * 
 * <pre> -F
 *  Weight neighbours by 1 - their distance (distance weighting is not supported yet)
 *  (use when k &gt; 1)</pre>
 * 
 * <pre> -K &lt;number of neighbors (base cases)&gt;
 *  Number of nearest neighbours (k) used in regression.
 *  (Default = 1)</pre>
 *
 *  <pre> -L &lt;number of adaptations per base case&gt;
 *  Number of adaptations per base case used in regression.
 *  (use when l &gt; 1)</pre>
 *
 *  <pre> -O &lt; scaling coefficient for R, used for defining the
 *  neighborhood for rule generation. For example, if R is set to
 *  10 and O is set to 1.5, the rules will be generated from the top
 *  15 nearest neighbors of the input query.
 *  (Default = 1)</pre>
 *
 *
 *  <pre> -M &lt;number of nearest neighbors for generating adaptations&gt;
 *  Number of nearest neighbors to use for generating adaptation.
 *  (use when m &gt; 1)</pre>
 * 
 * <pre> -E
 *  Minimise mean squared error rather than mean absolute
 *  error when using -X option with numeric prediction.</pre>
 * 
 * <pre> -W &lt;window size&gt;
 *  Maximum number of training instances maintained.
 *  Training instances are dropped FIFO. (Default = no window)</pre>
 * 
 * <pre> -X
 *  Select the number of nearest neighbours (base cases) between 1
 *  and the k value specified and the number of adaptations to apply per base case between
 *  1 and l value using hold-one-out evaluation
 *  on the training data (use when k and l &gt; 1)</pre>
 * 
 * <pre> -A
 *  The nearest neighbour search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
 * </pre>
 * 
* <pre> -B
 *  The rule retrieval search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
 * </pre>
 *
 <!-- options-end -->
 *
 * @author Vahid Jalali (vjalalib@cs.indiana.edu)
 * @version $Revision: 1.0 $
 * Adapted drom IBk implemented by
 * Stuart Inglis (singlis@cs.waikato.ac.nz)
 * Len Trigg (trigg@cs.waikato.ac.nz)
 * Eibe Frank (eibe@cs.waikato.ac.nz)
 *
 */
public class EAR4
  extends Classifier 
  implements OptionHandler, UpdateableClassifier, WeightedInstancesHandler,
             TechnicalInformationHandler, AdditionalMeasureProducer {

  /** for serialization. */
  static final long serialVersionUID = -3080186098777067173L;

  /** The training instances used for regression. */
  protected Instances m_Train;

  /** The number of class values (or 1 if predicting numeric). */
  protected int m_NumClasses;

  /** The class attribute type. */
  protected int m_ClassType;

  /** The number of neighbours (base cases) to use for regression (currently). */
  protected int m_kNN;

  /**
   * The value of kNN provided by the user. This may differ from
   * m_kNN if cross-validation is being used ( cross validation is not supported yet).
   */
  protected int m_kNNUpper;

  /** The number of adaptations per base case to use for regression. */
  protected int m_l;

  /** Rule generation neighborhood specifier scaler coefficient. */
  protected double m_o = 1;

  /**
   * The value of L provided by the user. This may differ from
   * m_l if cross-validation is being used. ( cross validation is not supported yet)
   */
  protected int m_lUpper;

  /**
   * Whether the value of k and l selected by cross validation has
   * been invalidated by a change in the training instances. ( cross validation is not supported yet)
   */
  protected boolean m_kNNValid;


  /**
   * The maximum number of training instances allowed. When
   * this limit is reached, old training instances are removed,
   * so the training data is "windowed". Set to 0 for unlimited
   * numbers of instances.
   */
  protected int m_WindowSize;

  /** Whether the neighbours should be distance-weighted. (distance weighting is not supported yet) */
  protected int m_DistanceWeighting;

  /** Whether to select k and l by cross validation. (cross validation is not supported yet) */
  protected boolean m_CrossValidate;

  /**
   * Whether to minimise mean squared error rather than mean absolute
   * error when cross-validating on numeric prediction tasks. (cross validation is not supported yet)
   */
  protected boolean m_MeanSquared;

  /** no weighting. */
  public static final int WEIGHT_NONE = 1; //(distance weighting is not supported yet)
  /** weight by 1/distance. */
  public static final int WEIGHT_INVERSE = 2;
  /** weight by 1-distance. */
  public static final int WEIGHT_SIMILARITY = 4;
  /** possible instance weighting methods. */
  public static final Tag [] TAGS_WEIGHTING = {
    new Tag(WEIGHT_NONE, "No distance weighting"),
    new Tag(WEIGHT_INVERSE, "Weight by 1/distance"),
    new Tag(WEIGHT_SIMILARITY, "Weight by 1-distance")
  };
  
  /** for nearest-neighbor search. */
  protected NearestNeighbourSearch m_NNSearch = new LinearNNSearch();

  /** for adaptation rule nearest-neighbor search. */
  protected NearestNeighbourSearch l_NNSearch = new LinearNNSearch();

  /** The number of attributes the contribute to a prediction. */
  protected double m_NumAttributesUsed;
  
  /** Default ZeroR model to use when there are no training instances */
  protected ZeroR m_defaultModel;
  
  /**
   * EAR4 learner. Case-based learner that uses ensembles of adaptations to adjust the value
   * of the nearetst k traning instances for predicting the target value of the test instance
   *
   * @param k the number of nearest neighbors (base cases) to use for prediction
   * @param l the number of adaptations to apply per base case to adjust its value
   */
  public EAR4(int k, int l) {

    init();
    setKNN(k);
	setl(l);
  }  

  /**
   * EAR4 learner. Case-based learner that uses ensembles of adaptations to adjust the value 
   * of the nearetst k traning instances for predicting the target value of the test instance 
   */
  public EAR4() {

    init();
  }
  
  /**
   * Returns a string describing classifier.
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    /*return  "EAR4 learner. Can "
      + "select appropriate value of K and l based on cross-validation. Can also do "
      + "distance weighting.\n\n" (distance weighting is not supported yet)
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();*/

    return  "EAR4 learner. "
      + "For more information, see\n\n"
      + getTechnicalInformation().toString();

  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.CONFERENCE);
    result.setValue(Field.AUTHOR, "V. Jalali and D. Leake");
    result.setValue(Field.YEAR, "2013");
    result.setValue(Field.TITLE, "Extending case adaptation with automatically-generated ensembles of adaptation rules");
    result.setValue(Field.BOOKTITLE, "Case-Based Reasoning Research and Development");
    result.setValue(Field.PAGES, "188-202");
	result.setValue(Field.PUBLISHER, "Springer");
    
    return result;
  }

  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String KNNTipText() {
    return "The number of neighbors to use.";
  }
  
  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String lTipText() {
    return "The number of adaptations to use per base case.";
  }

  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String oTipText() {
    return "The rule generation neighborhood selection scaling coefficient.";
  }

  /**
   * Set the number of neighbours the learner is to use.
   *
   * @param k the number of neighbours.
   */
  public void setKNN(int k) {
    m_kNN = k;
    m_kNNUpper = k;
    m_kNNValid = false;
  }

  /**
   * Set the number of adaptations to use per base case.
   *
   * @param l the number of adaptations.
   */
  public void setl(int l) {
    m_l = l;
    m_lUpper = l;
    m_kNNValid = false;
  }

  /**
   * Set the rule generation neighborhood specifier coefficient.
   *
   * @param o rule generation neighborhood specifier coefficient.
   */
  public void seto(double o) {
    m_o = o;
  }

  /**
   * Gets the number of neighbours the learner will use.
   *
   * @return the number of neighbours.
   */
  public int getKNN() {

    return m_kNN;
  }

  /**
   * gets the number of adaptations to apply per base case.
   *
   * @return the number of adaptations to use per base case.
   */
  public int getl() {

    return m_l;
  }

  /**
   * gets the rule generation neighborhood specifier coefficient.
   *
   * @return the rule generation neighborhood specifier coefficient.
   */
  public double geto() {

    return m_o;
  }


  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String windowSizeTipText() {
    return "Gets the maximum number of instances allowed in the training " +
      "pool. The addition of new instances above this value will result " +
      "in old instances being removed. A value of 0 signifies no limit " +
      "to the number of training instances.";
  }
  
  /**
   * Gets the maximum number of instances allowed in the training
   * pool. The addition of new instances above this value will result
   * in old instances being removed. A value of 0 signifies no limit
   * to the number of training instances.
   *
   * @return Value of WindowSize.
   */
  public int getWindowSize() {
    
    return m_WindowSize;
  }
  
  /**
   * Sets the maximum number of instances allowed in the training
   * pool. The addition of new instances above this value will result
   * in old instances being removed. A value of 0 signifies no limit
   * to the number of training instances.
   *
   * @param newWindowSize Value to assign to WindowSize.
   */
  public void setWindowSize(int newWindowSize) {
    
    m_WindowSize = newWindowSize;
  }
  
  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui (distance weighting is not supported yet)
   */
  /*public String distanceWeightingTipText() {

    return "Gets the distance weighting method used.";
  }*/
  
  /**
   * Gets the distance weighting method used. Will be one of
   * WEIGHT_NONE, WEIGHT_INVERSE, or WEIGHT_SIMILARITY (distance weighting is not supported yet)
   *
   * @return the distance weighting method used.
   */
  /*public SelectedTag getDistanceWeighting() {

    return new SelectedTag(m_DistanceWeighting, TAGS_WEIGHTING);
  }*/
  
  /**
   * Sets the distance weighting method used. Values other than
   * WEIGHT_NONE, WEIGHT_INVERSE, or WEIGHT_SIMILARITY will be ignored. (distance weighting is not supported yet)
   *
   * @param newMethod the distance weighting method to use
   */
  /*public void setDistanceWeighting(SelectedTag newMethod) {
    
    if (newMethod.getTags() == TAGS_WEIGHTING) {
      m_DistanceWeighting = newMethod.getSelectedTag().getID();
    }
  }*/
  
  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  /*public String meanSquaredTipText() {

    return "Whether the mean squared error is used rather than mean "
      + "absolute error when doing cross-validation for regression problems. ( cross validation is not supported yet)";
  }*/

  /**
   * Gets whether the mean squared error is used rather than mean
   * absolute error when doing cross-validation. ( cross validation is not supported yet)
   *
   * @return true if so.
   */
  /*public boolean getMeanSquared() {
    
    return m_MeanSquared;
  }*/
  
  /**
   * Sets whether the mean squared error is used rather than mean
   * absolute error when doing cross-validation. ( cross validation is not supported yet)
   *
   * @param newMeanSquared true if so.
   */
  /*public void setMeanSquared(boolean newMeanSquared) {
    
    m_MeanSquared = newMeanSquared;
  }*/
  
  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui ( cross validation is not supported yet)
   */
  /*public String crossValidateTipText() {

    return "Whether hold-one-out cross-validation will be used to " +
      "select the best k value between 1 and the value specified as " +
      "the KNN parameter.";
  }*/
  
  /**
   * Gets whether hold-one-out cross-validation will be used
   * to select the best k value. ( cross validation is not supported yet)
   *
   * @return true if cross-validation will be used.
   */
  /*public boolean getCrossValidate() {
    
    return m_CrossValidate;
  }*/
  
  /**
   * Sets whether hold-one-out cross-validation will be used
   * to select the best k value. ( cross validation is not supported yet)
   *
   * @param newCrossValidate true if cross-validation should be used.
   */
  /*public void setCrossValidate(boolean newCrossValidate) {
    
    m_CrossValidate = newCrossValidate;
  }*/

  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String nearestNeighbourSearchAlgorithmTipText() {
    return "The nearest neighbour search algorithm to use " +
    	   "(Default: weka.core.neighboursearch.LinearNNSearch).";
  }
 
  /**
   * Returns the tip text for this property.
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String ruleNearestNeighbourSearchAlgorithmTipText() {
    return "The rule retrieval algorithm to use " +
    	   "(Default: weka.core.neighboursearch.LinearNNSearch).";
  }

  /**
   * Returns the current nearestNeighbourSearch algorithm in use.
   * @return the NearestNeighbourSearch algorithm currently in use.
   */
  public NearestNeighbourSearch getNearestNeighbourSearchAlgorithm() {
    return m_NNSearch;
  }
  
  /**
   * Sets the nearestNeighbourSearch algorithm to be used for finding nearest
   * neighbour(s).
   * @param nearestNeighbourSearchAlgorithm - The NearestNeighbourSearch class.
   */
  public void setNearestNeighbourSearchAlgorithm(NearestNeighbourSearch nearestNeighbourSearchAlgorithm) {
    m_NNSearch = nearestNeighbourSearchAlgorithm;
  }
 
  /**
   * Returns the current nearestNeighbourSearch algorithm (for rule retrieval) in use.
   * @return the NearestNeighbourSearch algorithm currently in use.
   */
  public NearestNeighbourSearch getRuleNearestNeighbourSearchAlgorithm() {
    return l_NNSearch;
  }
  
  /**
   * Sets the nearestNeighbourSearch algorithm to be used for rule retrieval
   * @param nearestNeighbourSearchAlgorithm - The NearestNeighbourSearch class.
   */
  public void setRuleNearestNeighbourSearchAlgorithm(NearestNeighbourSearch nearestNeighbourSearchAlgorithm) {
    l_NNSearch = nearestNeighbourSearchAlgorithm;
  }
  
  /**
   * Get the number of training instances the classifier is currently using.
   * 
   * @return the number of training instances the classifier is currently using
   */
  public int getNumTraining() {

    return m_Train.numInstances();
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    //result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    //result.enable(Capability.DATE_ATTRIBUTES);
    //result.enable(Capability.MISSING_VALUES);

    // class
    //result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    //result.enable(Capability.DATE_CLASS);
    //result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }
  
  /**
   * Generates the classifier.
   *
   * @param instances set of instances serving as training data 
   * @throws Exception if the classifier has not been generated successfully
   */
  public void buildClassifier(Instances instances) throws Exception {
    
    // can classifier handle the data?
    getCapabilities().testWithFail(instances);

    // remove instances with missing class
    instances = new Instances(instances);
    instances.deleteWithMissingClass();
    //inja add filtering for non numerical attributes

    m_NumClasses = instances.numClasses();
    m_ClassType = instances.classAttribute().type();
    m_Train = new Instances(instances, 0, instances.numInstances());

    // Throw away initial instances until within the specified window size
    if ((m_WindowSize > 0) && (instances.numInstances() > m_WindowSize)) {
      m_Train = new Instances(m_Train, 
			      m_Train.numInstances()-m_WindowSize, 
			      m_WindowSize);
    }

    m_NumAttributesUsed = 0.0;
    
	for (int i = 0; i < m_Train.numAttributes(); i++) {
      if ((i != m_Train.classIndex()) && 
	  (m_Train.attribute(i).isNominal() ||
	   m_Train.attribute(i).isNumeric())) {
	m_NumAttributesUsed += 1.0;
      }
    }

   
    m_NNSearch.setInstances(m_Train);

    // Invalidate any currently cross-validation selected k
    m_kNNValid = false;
    
    m_defaultModel = new ZeroR();
    m_defaultModel.buildClassifier(instances);
  }

  /**
   * Adds the supplied instance to the training set.
   *
   * @param instance the instance to add
   * @throws Exception if instance could not be incorporated
   * successfully
   */
  public void updateClassifier(Instance instance) throws Exception {

    if (m_Train.equalHeaders(instance.dataset()) == false) {
      throw new Exception("Incompatible instance types");
    }
    if (instance.classIsMissing()) {
      return;
    }

    m_Train.add(instance);
    m_NNSearch.update(instance);
    m_kNNValid = false;
    if ((m_WindowSize > 0) && (m_Train.numInstances() > m_WindowSize)) {
      boolean deletedInstance=false;
      while (m_Train.numInstances() > m_WindowSize) {
	m_Train.delete(0);
        deletedInstance=true;
      }
      //rebuild datastructure KDTree currently can't delete
      if(deletedInstance==true)
        m_NNSearch.setInstances(m_Train);
    }
  }

  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @throws Exception if an error occurred during the prediction
   */
  public double [] distributionForInstance(Instance instance) throws Exception {
	if (m_kNN * m_o > m_Train.numInstances() * (m_Train.numInstances() - 1))
	{
		//if m_kNN * m_o is larger than the maximum possible or rules, set m_o to the maximum feasible value
		m_o = (int) java.lang.Math.round(m_Train.numInstances() * (m_Train.numInstances() - 1)/m_kNN);
	}
    if (m_Train.numInstances() == 0) {
      //throw new Exception("No training instances!");
      return m_defaultModel.distributionForInstance(instance);
    }
    if ((m_WindowSize > 0) && (m_Train.numInstances() > m_WindowSize)) {
      m_kNNValid = false;
      boolean deletedInstance=false;
      while (m_Train.numInstances() > m_WindowSize) {
	m_Train.delete(0);
      }
      //rebuild datastructure KDTree currently can't delete
      if(deletedInstance==true)
        m_NNSearch.setInstances(m_Train);
    }

    // Select k by cross validation ( cross validation is not supported yet)
    //if (!m_kNNValid && (m_CrossValidate) && (m_kNNUpper >= 1)) {
    //	crossValidate();
    //}

    m_NNSearch.addInstanceInfo(instance);

    Instances neighbours = m_NNSearch.kNearestNeighbours(instance, (int) java.lang.Math.round(m_kNN * m_o));
	Instances rules = generateRules(neighbours);
	rules.setClassIndex(rules.numAttributes() -1);
	while(neighbours.numInstances() > m_kNN)
		neighbours.delete(neighbours.numInstances() - 1);
	double  prediction = predictValue(instance, neighbours, rules);
	return new double[]{prediction};
    //double [] distances = m_NNSearch.getDistances();
    //double [] distribution = makeDistribution( neighbours, distances);

    //return distribution;
  }

	private double predictValue(Instance target, Instances neighbors, Instances rules) throws Exception
	{
		l_NNSearch.setInstances(rules);

		double prediction = 0;
		for(int i =0; i < neighbors.numInstances(); i++)
		{
			double adjustment = 0;
			if (m_l > 0)
			{
				Instance diffToAddress = generateRule(target, neighbors.instance(i));
				Instances rulesToApply = l_NNSearch.kNearestNeighbours(diffToAddress, m_l);
				for (int j = 0; j < rulesToApply.numInstances(); j++)
				{
					adjustment += rulesToApply.instance(j).value(rulesToApply.instance(j).classAttribute());
				}
		
				adjustment = adjustment / m_l;
			}
			prediction += neighbors.instance(i).value(neighbors.instance(i).classAttribute()) + adjustment;
			//System.out.println(adjustment + " " + neighbors.instance(i).value(neighbors.instance(i).classAttribute()));
		}
		prediction = prediction / m_kNN;
		return prediction;
	}

	private Instances generateRules(Instances baseCases)
	{
		FastVector newAttributes = new FastVector();
		//System.out.println("number of attrs " + baseCases.firstInstance().numAttributes() );
    	for (int i = 0; i < baseCases.firstInstance().numAttributes(); i++) {
			newAttributes.addElement(baseCases.firstInstance().attribute(i));
		}
		Instances rules = new Instances("rules", newAttributes, baseCases.numInstances() * baseCases.numInstances() );
		for (int i = 0; i < baseCases.numInstances(); i++)
		{
			for (int j = 0; j < baseCases.numInstances(); j++)
			{
				if (i != j)
					rules.add(generateRule(baseCases.instance(i), baseCases.instance(j)));
			}
		}
		return rules;
	}

	private Instance generateRule(Instance case1, Instance case2)
	{
		Instance tempInst = (Instance)case1.copy();
		for (int i = 0; i < tempInst.numAttributes(); i++)
		{
			tempInst.setValue(i, tempInst.value(i) - case2.value(i));
			//System.out.print(tempInst.value(i) + " " +  case2.value(i) +  ",     ");
		}

		/*java.util.Enumeration<Attribute> attrs = tempInst.enumerateAttributes();
		System.out.println(case1.toString() + " before +++++++++++++++++++++ ");
		while(attrs.hasMoreElements())
		{
			Attribute attr = (Attribute) attrs.nextElement();
			System.out.print(tempInst.value(attr) + " " +  case2.value(attr) +  ",     ");
			tempInst.setValue(attr, tempInst.value(attr) - case2.value(attr));
		}*/
		//System.out.println("----------------------------------------------"+ case1.toString());
		return tempInst;
	}

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

	//inja make sure if it should be 8 rather than 9
    Vector newVector = new Vector(8);

    /*newVector.addElement(new Option(
	      "\tWeight neighbours by the inverse of their distance\n"+
	      "\t(use when k > 1)",
	      "I", 0, "-I"));
    newVector.addElement(new Option(
	      "\tWeight neighbours by 1 - their distance\n"+
	      "\t(use when k > 1)",
	      "F", 0, "-F"));*/
    newVector.addElement(new Option(
	      "\tNumber of nearest neighbours (k) used in classification.\n"+
	      "\t(Default = 1)",
	      "K", 1,"-K <number of neighbors>"));
    newVector.addElement(new Option(
	      "\tNumber of adaptations per base case (l) used in regression.\n"+
	      "\t(Default = 1)",
	      "L", 1,"-L <number of adaptations per base case>"));
    newVector.addElement(new Option(
          "\tThe neighborhood specifier coefficient for generating the adaptation rules.\n"+
          "\t(Default = 1)",
          "O", 1,"-O <rule generation neighborhood specifier coefficient>"));
    newVector.addElement(new Option(
          "\tMinimise mean squared error rather than mean absolute\n"+
	      "\terror when using -X option with numeric prediction.",
	      "E", 0,"-E"));
    newVector.addElement(new Option(
          "\tMaximum number of training instances maintained.\n"+
	      "\tTraining instances are dropped FIFO. (Default = no window)",
	      "W", 1,"-W <window size>"));
    newVector.addElement(new Option(
	      "\tSelect the number of nearest neighbours between 1\n"+
	      "\tand the k value specified using hold-one-out evaluation\n"+
	      "\ton the training data (use when k > 1)",
	      "X", 0,"-X"));
    newVector.addElement(new Option(
	      "\tThe nearest neighbour search algorithm to use "+
          "(default: weka.core.neighboursearch.LinearNNSearch).\n",
	      "A", 0, "-A"));
    newVector.addElement(new Option(
	      "\tThe rule retrieval algorithm to use "+
          "(default: weka.core.neighboursearch.LinearNNSearch).\n",
	      "B", 0, "-B"));

    return newVector.elements();
  }

   /* <pre> -I
     Weight neighbours by the inverse of their distance
     (use when k &gt; 1)</pre>
    
    <pre> -F
     Weight neighbours by 1 - their distance
     (use when k &gt; 1)</pre>*/

  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -K &lt;number of neighbors&gt;
   *  Number of nearest neighbours (k) used in classification.
   *  (Default = 1)</pre>
   *
   * <pre> -L &lt;number of adaptations per base case&gt;
   *  Number of applied adaptations per base case (l) used in regression.
   *  (Default = 1)</pre>
   *
   * <pre> -O &lt;rule generation neighborhood specifier coefficient&gt;
   *  coefficient for scalign the number of nearest neighbors for definging the neighborhood
   *  from which adaptation rules will be generated.
   *  (Default = 1)</pre>
   * 
   * <pre> -E
   *  Minimise mean squared error rather than mean absolute
   *  error when using -X option with numeric prediction.</pre>
   * 
   * <pre> -W &lt;window size&gt;
   *  Maximum number of training instances maintained.
   *  Training instances are dropped FIFO. (Default = no window)</pre>
   * 
   * <pre> -X
   *  Select the number of nearest neighbours between 1
   *  and the k value specified using hold-one-out evaluation
   *  on the training data (use when k &gt; 1)</pre>
   * 
   * <pre> -A
   *  The nearest neighbour search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
   * </pre>
   * 
   * <pre> -B
   *  The rule retrieval algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
   * </pre>
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    
    String knnString = Utils.getOption('K', options);
    if (knnString.length() != 0) {
      setKNN(Integer.parseInt(knnString));
    } else {
      setKNN(1);
    }
	String lString = Utils.getOption('L', options);
	if (lString.length() != 0) {
      setl(Integer.parseInt(lString));
    } else {
      setl(1);
    }
	String oString = Utils.getOption('O', options);
	if (oString.length() != 0) {
      seto(Double.parseDouble(oString));
    } else {
      seto(1);
    }
    String windowString = Utils.getOption('W', options);
    if (windowString.length() != 0) {
      setWindowSize(Integer.parseInt(windowString));
    } else {
      setWindowSize(0);
    }
    /*if (Utils.getFlag('I', options)) {
      setDistanceWeighting(new SelectedTag(WEIGHT_INVERSE, TAGS_WEIGHTING));
    } else if (Utils.getFlag('F', options)) {
      setDistanceWeighting(new SelectedTag(WEIGHT_SIMILARITY, TAGS_WEIGHTING));
    } else {
      setDistanceWeighting(new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING));
    }
    setCrossValidate(Utils.getFlag('X', options));
    setMeanSquared(Utils.getFlag('E', options));*/

    String nnSearchClass = Utils.getOption('A', options);
    if(nnSearchClass.length() != 0) {
      String nnSearchClassSpec[] = Utils.splitOptions(nnSearchClass);
      if(nnSearchClassSpec.length == 0) { 
        throw new Exception("Invalid NearestNeighbourSearch algorithm " +
                            "specification string."); 
      }
      String className = nnSearchClassSpec[0];
      nnSearchClassSpec[0] = "";

      setNearestNeighbourSearchAlgorithm( (NearestNeighbourSearch)
                  Utils.forName( NearestNeighbourSearch.class, 
                                 className, 
                                 nnSearchClassSpec)
                                        );
    }
    else 
      this.setNearestNeighbourSearchAlgorithm(new LinearNNSearch());

    String lSearchClass = Utils.getOption('B', options);
    if(lSearchClass.length() != 0) {
      String lSearchClassSpec[] = Utils.splitOptions(lSearchClass);
      if(lSearchClassSpec.length == 0) { 
        throw new Exception("Invalid NearestNeighbourSearch algorithm " +
                            "specification string."); 
      }
      String className = lSearchClassSpec[0];
      lSearchClassSpec[0] = "";

      setRuleNearestNeighbourSearchAlgorithm( (NearestNeighbourSearch)
                  Utils.forName( NearestNeighbourSearch.class, 
                                 className, 
                                 lSearchClassSpec)
                                        );
    }
    else 
      this.setRuleNearestNeighbourSearchAlgorithm(new LinearNNSearch());
   
    Utils.checkForRemainingOptions(options);
  }

  /**
   * Gets the current settings of IBk.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String [] getOptions() {

    String [] options = new String [12];
    int current = 0;
    options[current++] = "-K"; options[current++] = "" + getKNN();
	options[current++] = "-L"; options[current++] = "" + getl();
	options[current++] = "-O"; options[current++] = "" + geto();
    options[current++] = "-W"; options[current++] = "" + m_WindowSize;
    /*if (getCrossValidate()) {
      options[current++] = "-X";
    }
    if (getMeanSquared()) {
      options[current++] = "-E";
    }
    if (m_DistanceWeighting == WEIGHT_INVERSE) {
      options[current++] = "-I";
    } else if (m_DistanceWeighting == WEIGHT_SIMILARITY) {
      options[current++] = "-F";
    }*/

    options[current++] = "-A";
    options[current++] = m_NNSearch.getClass().getName()+" "+Utils.joinOptions(m_NNSearch.getOptions()); 
   
	options[current++] = "-B";
    options[current++] = l_NNSearch.getClass().getName() +" "+Utils.joinOptions(l_NNSearch.getOptions()); 
    
	while (current < options.length) {
      options[current++] = "";
    }
    
    return options;
  }

  /**
   * Returns an enumeration of the additional measure names 
   * produced by the neighbour search algorithm, plus the chosen K in case
   * cross-validation is enabled. (cross validation is not supported yet)
   * 
   * @return an enumeration of the measure names
   */
  public Enumeration enumerateMeasures() {
    if (m_CrossValidate) {
      Enumeration enm = m_NNSearch.enumerateMeasures();
      Vector measures = new Vector();
      while (enm.hasMoreElements())
	measures.add(enm.nextElement());
      measures.add("measureKNN");
      return measures.elements();
    }
    else {
      return m_NNSearch.enumerateMeasures();
    }
  }
  
  /**
   * Returns the value of the named measure from the 
   * neighbour search algorithm, plus the chosen K in case
   * cross-validation is enabled. (cross validation is not supported yet)
   * 
   * @param additionalMeasureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  public double getMeasure(String additionalMeasureName) {
    if (additionalMeasureName.equals("measureKNN"))
      return m_kNN;
    else
      return m_NNSearch.getMeasure(additionalMeasureName);
  }
  
  
  /**
   * Returns a description of this classifier.
   *
   * @return a description of this classifier as a string.
   */
  public String toString() {

    if (m_Train == null) {
      return "IBk: No model built yet.";
    }
    
    if (m_Train.numInstances() == 0) {
      return "Warning: no training instances - ZeroR model used.";
    }    

	//( cross validation is not supported yet)
    //if (!m_kNNValid && m_CrossValidate) {
    //  crossValidate();
    //}
    
    

    String result = "EAR4 learner\n" +
      "using " + m_kNN;

    /*switch (m_DistanceWeighting) {
    case WEIGHT_INVERSE:
      result += " inverse-distance-weighted";
      break;
    case WEIGHT_SIMILARITY:
      result += " similarity-weighted";
      break;
    }*/
    result += " nearest neighbour(s) and " + m_l + " adptations per base case and " + m_o + " as the rule generation neighborhood selection" +
				" scaling coefficient for predicting case values.\n";

    if (m_WindowSize != 0) {
      result += "using a maximum of " 
	+ m_WindowSize + " (windowed) training instances\n";
    }
    return result;
  }

  /**
   * Initialise scheme variables.
   */
  protected void init() {

    setKNN(1);
    m_WindowSize = 0;
    m_DistanceWeighting = WEIGHT_NONE;
    m_CrossValidate = false;
    m_MeanSquared = false;
  }
  
  /**
   * Turn the list of nearest neighbors into a probability distribution.
   *
   * @param neighbours the list of nearest neighboring instances
   * @param distances the distances of the neighbors
   * @return the probability distribution
   * @throws Exception if computation goes wrong or has no class attribute
   */
  protected double [] makeDistribution(Instances neighbours, double[] distances)
    throws Exception {

    double total = 0, weight;
    double [] distribution = new double [m_NumClasses];
    
    // Set up a correction to the estimator
    if (m_ClassType == Attribute.NOMINAL) {
      for(int i = 0; i < m_NumClasses; i++) {
	distribution[i] = 1.0 / Math.max(1,m_Train.numInstances());
      }
      total = (double)m_NumClasses / Math.max(1,m_Train.numInstances());
    }

    for(int i=0; i < neighbours.numInstances(); i++) {
      // Collect class counts
      Instance current = neighbours.instance(i);
      distances[i] = distances[i]*distances[i];
      distances[i] = Math.sqrt(distances[i]/m_NumAttributesUsed);
      switch (m_DistanceWeighting) {
        case WEIGHT_INVERSE:
          weight = 1.0 / (distances[i] + 0.001); // to avoid div by zero
          break;
        case WEIGHT_SIMILARITY:
          weight = 1.0 - distances[i];
          break;
        default:                                 // WEIGHT_NONE:
          weight = 1.0;
          break;
      }
      weight *= current.weight();
      try {
        switch (m_ClassType) {
          case Attribute.NOMINAL:
            distribution[(int)current.classValue()] += weight;
            break;
          case Attribute.NUMERIC:
            distribution[0] += current.classValue() * weight;
            break;
        }
      } catch (Exception ex) {
        throw new Error("Data has no class attribute!");
      }
      total += weight;      
    }

    // Normalise distribution
    if (total > 0) {
      Utils.normalize(distribution, total);
    }
    return distribution;
  }

  /**
   * Select the best value for k by hold-one-out cross-validation. ( cross validation is not supported yet)
   * If the class attribute is nominal, classification error is
   * minimised. If the class attribute is numeric, mean absolute
   * error is minimised
   */
  protected void crossValidate() {

    try {
      if (m_NNSearch instanceof weka.core.neighboursearch.CoverTree)
	throw new Exception("CoverTree doesn't support hold-one-out "+
			    "cross-validation. Use some other NN " +
			    "method.");

      double [] performanceStats = new double [m_kNNUpper];
      double [] performanceStatsSq = new double [m_kNNUpper];

      for(int i = 0; i < m_kNNUpper; i++) {
	performanceStats[i] = 0;
	performanceStatsSq[i] = 0;
      }


      m_kNN = m_kNNUpper;
      Instance instance;
      Instances neighbours;
      double[] origDistances, convertedDistances;
      for(int i = 0; i < m_Train.numInstances(); i++) {
	if (m_Debug && (i % 50 == 0)) {
	  System.err.print("Cross validating "
			   + i + "/" + m_Train.numInstances() + "\r");
	}
	instance = m_Train.instance(i);
	neighbours = m_NNSearch.kNearestNeighbours(instance, m_kNN);
        origDistances = m_NNSearch.getDistances();
        
	for(int j = m_kNNUpper - 1; j >= 0; j--) {
	  // Update the performance stats
          convertedDistances = new double[origDistances.length];
          System.arraycopy(origDistances, 0, 
                           convertedDistances, 0, origDistances.length);
	  double [] distribution = makeDistribution(neighbours, 
                                                    convertedDistances);
          double thisPrediction = Utils.maxIndex(distribution);
	  if (m_Train.classAttribute().isNumeric()) {
	    thisPrediction = distribution[0];
	    double err = thisPrediction - instance.classValue();
	    performanceStatsSq[j] += err * err;   // Squared error
	    performanceStats[j] += Math.abs(err); // Absolute error
	  } else {
	    if (thisPrediction != instance.classValue()) {
	      performanceStats[j] ++;             // Classification error
	    }
	  }
	  if (j >= 1) {
	    neighbours = pruneToK(neighbours, convertedDistances, j);
	  }
	}
      }

      // Display the results of the cross-validation
      for(int i = 0; i < m_kNNUpper; i++) {
	if (m_Debug) {
	  System.err.print("Hold-one-out performance of " + (i + 1)
			   + " neighbors " );
	}
	if (m_Train.classAttribute().isNumeric()) {
	  if (m_Debug) {
	    if (m_MeanSquared) {
	      System.err.println("(RMSE) = "
				 + Math.sqrt(performanceStatsSq[i]
					     / m_Train.numInstances()));
	    } else {
	      System.err.println("(MAE) = "
				 + performanceStats[i]
				 / m_Train.numInstances());
	    }
	  }
	} else {
	  if (m_Debug) {
	    System.err.println("(%ERR) = "
			       + 100.0 * performanceStats[i]
			       / m_Train.numInstances());
	  }
	}
      }


      // Check through the performance stats and select the best
      // k value (or the lowest k if more than one best)
      double [] searchStats = performanceStats;
      if (m_Train.classAttribute().isNumeric() && m_MeanSquared) {
	searchStats = performanceStatsSq;
      }
      double bestPerformance = Double.NaN;
      int bestK = 1;
      for(int i = 0; i < m_kNNUpper; i++) {
	if (Double.isNaN(bestPerformance)
	    || (bestPerformance > searchStats[i])) {
	  bestPerformance = searchStats[i];
	  bestK = i + 1;
	}
      }
      m_kNN = bestK;
      if (m_Debug) {
	System.err.println("Selected k = " + bestK);
      }
      
      m_kNNValid = true;
    } catch (Exception ex) {
      throw new Error("Couldn't optimize by cross-validation: "
		      +ex.getMessage());
    }
  }
  
  /**
   * Prunes the list to contain the k nearest neighbors. If there are
   * multiple neighbors at the k'th distance, all will be kept.
   *
   * @param neighbours the neighbour instances.
   * @param distances the distances of the neighbours from target instance.
   * @param k the number of neighbors to keep.
   * @return the pruned neighbours.
   */
  public Instances pruneToK(Instances neighbours, double[] distances, int k) {
    
    if(neighbours==null || distances==null || neighbours.numInstances()==0) {
      return null;
    }
    if (k < 1) {
      k = 1;
    }
    
    int currentK = 0;
    double currentDist;
    for(int i=0; i < neighbours.numInstances(); i++) {
      currentK++;
      currentDist = distances[i];
      if(currentK>k && currentDist!=distances[i-1]) {
        currentK--;
        neighbours = new Instances(neighbours, 0, currentK);
        break;
      }
    }

    return neighbours;
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1.0 $");
  }
  
  /**
   * Main method for testing this class.
   *
   * @param argv should contain command line options (see setOptions)
   */
  public static void main(String [] argv) {
    runClassifier(new EAR4(), argv);
  }
}
